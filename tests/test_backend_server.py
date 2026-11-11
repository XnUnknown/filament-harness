import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from filament import __version__, backend_server
from filament.chat import ChatOrchestrator
from filament.config import HarnessConfig
from filament.llm import MockLLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_backend(payload: bytes, cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run the backend with a non-UTF-8 locale to exercise pipe framing."""
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT), env.get("PYTHONPATH", "")) if part
    )
    env["PYTHONIOENCODING"] = "cp1252"
    return subprocess.run(
        [sys.executable, "-m", "filament.backend_server"],
        input=payload,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


class TestBackendProtocol(unittest.TestCase):
    def test_ping_reports_package_version(self):
        reply = json.loads(backend_server.respond('{"id": 1, "method": "ping"}'))
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["id"], 1)
        self.assertEqual(reply["result"]["version"], __version__)

    def test_bad_line_returns_error_instead_of_raising(self):
        reply = json.loads(backend_server.respond("not json"))
        self.assertFalse(reply["ok"])
        self.assertIsNone(reply["id"])
        self.assertTrue(reply["error"])

    def test_rejects_non_object_request(self):
        reply = json.loads(backend_server.respond("[1, 2]"))
        self.assertFalse(reply["ok"])
        self.assertIn("JSON object", reply["error"])

    def test_rejects_non_object_params(self):
        reply = json.loads(backend_server.respond('{"id": 3, "method": "ping", "params": []}'))
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["id"], 3)
        self.assertIn("params", reply["error"])

    def test_rejects_non_string_method(self):
        reply = json.loads(backend_server.respond('{"id": 4, "method": 7}'))
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["id"], 4)
        self.assertIn("method", reply["error"])

    def test_unknown_method_is_an_error(self):
        reply = json.loads(backend_server.respond('{"id": 5, "method": "explode"}'))
        self.assertFalse(reply["ok"])
        self.assertIn("unknown method: explode", reply["error"])

    def test_unserializable_result_keeps_request_id(self):
        with patch.object(backend_server, "handle", return_value=object()):
            reply = json.loads(backend_server.respond('{"id": 6, "method": "ping"}'))
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["id"], 6)
        self.assertIn("serializable", reply["error"])


class TestBackendProcess(unittest.TestCase):
    def test_utf8_framing_survives_cp1252_locale(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "工作区"
            workspace.mkdir()
            proc = run_backend(b'{"id": 1, "method": "ping"}\n', workspace)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            self.assertEqual(proc.stderr, b"")
            reply = json.loads(proc.stdout.decode("utf-8"))
            self.assertTrue(reply["ok"])
            self.assertEqual(reply["result"]["version"], __version__)
            self.assertEqual(Path(reply["result"]["workspace"]).resolve(), workspace.resolve())

    def test_bad_line_does_not_end_the_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = b'not json\n{"id": 2, "method": "ping"}\n'
            proc = run_backend(payload, Path(tmp))
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            replies = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines()]
            self.assertEqual(len(replies), 2)
            self.assertFalse(replies[0]["ok"])
            self.assertIsNone(replies[0]["id"])
            self.assertTrue(replies[1]["ok"])
            self.assertEqual(replies[1]["id"], 2)

    def test_oversized_line_is_rejected_and_session_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            padding = b"x" * backend_server.MAX_REQUEST_LINE_BYTES
            oversized = b'{"id": 3, "method": "ping", "params": {"pad": "' + padding + b'"}}\n'
            payload = oversized + b'{"id": 4, "method": "ping"}\n'
            proc = run_backend(payload, Path(tmp))
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            replies = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines()]
            self.assertEqual(len(replies), 2)
            self.assertFalse(replies[0]["ok"])
            self.assertIn("exceeds", replies[0]["error"])
            self.assertTrue(replies[1]["ok"])
            self.assertEqual(replies[1]["id"], 4)


class TestApplyConfig(unittest.TestCase):
    def setUp(self):
        self.config = HarnessConfig()

    def test_returns_validated_copy_and_leaves_original_untouched(self):
        updated = backend_server._apply_config(
            self.config, {"workers": 2, "llm": {"kind": "mock"}}
        )
        self.assertIsNot(updated, self.config)
        self.assertIsNot(updated.llm, self.config.llm)
        self.assertEqual(updated.workers, 2)
        self.assertEqual(updated.llm.kind, "mock")
        self.assertEqual(self.config.workers, 4)
        self.assertEqual(self.config.llm.kind, "auto")

    def test_failed_validation_leaves_live_config_untouched(self):
        with self.assertRaises(ValueError):
            backend_server._apply_config(self.config, {"workers": 2, "timeout_shell": 0})
        self.assertEqual(self.config.workers, 4)
        self.assertEqual(self.config.timeout_shell, 60)

    def test_rejects_wrong_types_and_unknown_keys(self):
        cases = [
            {"workers": "4"},
            {"max_tokens": True},
            {"budget_margin": "0.1"},
            {"atomic_writes": "yes"},
            {"command_allowlist": "ls"},
            {"unknown_key": 1},
            {"llm": "mock"},
            {"llm": {"unknown_key": "x"}},
            {"llm": {"timeout": "120"}},
            {"llm": {"kind": 5}},
        ]
        for updates in cases:
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    backend_server._apply_config(self.config, updates)
        self.assertEqual(self.config.workers, 4)

    def test_rejects_non_object_updates(self):
        with self.assertRaises(ValueError):
            backend_server._apply_config(self.config, ["workers"])

    def test_apply_config_rebuilds_client_and_toolbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = ChatOrchestrator(HarnessConfig(), tmp, mode="auto")
            updated = backend_server._apply_config(
                chat.config, {"llm": {"kind": "mock", "model": "test-model"}}
            )
            chat.apply_config(updated)
            self.assertIs(chat.config, updated)
            self.assertIs(chat.toolbox.config, updated)
            self.assertIsInstance(chat.client, MockLLM)

    def test_set_config_is_atomic_and_publishes_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = ChatOrchestrator(HarnessConfig(), tmp, mode="auto")
            backend_server.STATE.update({"workspace": Path(tmp), "config": chat.config, "chat": chat})
            try:
                updated = backend_server.handle("set_config", {"config": {"workers": 2}})
                self.assertEqual(updated["workers"], 2)
                self.assertIs(chat.config, backend_server.STATE["config"])

                with self.assertRaises(ValueError):
                    backend_server.handle("set_config", {"config": {"workers": "eight"}})
                with self.assertRaises(ValueError):
                    backend_server.handle("set_config", {"config": {"bogus_key": 1}})
                self.assertEqual(backend_server.handle("get_config", {})["workers"], 2)
            finally:
                backend_server.STATE.update({"workspace": None, "config": None, "chat": None})


if __name__ == "__main__":
    unittest.main()
