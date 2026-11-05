import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.context import ContextBudget
from filament.files import extract_code, ensure_target_within_root
from filament.llm import MockLLM, OllamaClient
from filament.tools import Toolset


class TestEdgeCases(unittest.TestCase):
    def test_extract_code_handles_crlf_fence(self):
        text = "\r\nprint(1)\n\r\n".replace(chr(92) + "r", chr(13))
        self.assertEqual(extract_code(text), "print(1)" + chr(10))

    def test_spawn_agent_refuses_when_budget_full(self):
        cfg = HarnessConfig(max_tokens=1024)
        budget = ContextBudget(max_tokens=cfg.effective_max_tokens, chars_per_token=4)
        budget.add("a" * (cfg.effective_max_tokens * 4))
        toolset = Toolset(client=MockLLM(), budget=budget, config=cfg, shell_cwd=".")
        with self.assertRaises(RuntimeError):
            toolset.spawn_agent("Write a Python function named add(a, b).")

    def test_replace_line_roundtrip_no_trailing_newline(self):
        from filament.files import replace_line
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.py"
            path.write_text("old", encoding="utf-8")
            replace_line(path, 1, "new")
            self.assertEqual(path.read_text(encoding="utf-8"), "new")

    def test_flatten_skips_disabled_subtasks(self):
        from filament.tasks import flatten_tasks, parse_tasks

        tasks = parse_tasks(
            [
                {
                    "id": "parent",
                    "task": "parent work",
                    "subtasks": [
                        {"id": "on", "task": "enabled", "enabled": True},
                        {"id": "off", "task": "disabled", "enabled": False},
                    ],
                }
            ]
        )
        self.assertEqual([task.id for task in flatten_tasks(tasks)], ["on"])

    def test_toolbox_run_shell_honors_allowlist(self):
        from filament.agent_tools import Toolbox

        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig(use_shell=False, command_allowlist=["echo"])
            tools = Toolbox(d, cfg)
            result = tools.run_shell("echo hello")
            self.assertIn("exit_code: 0", result)
            self.assertIn("hello", result)
            with self.assertRaises(ValueError):
                tools.run_shell("printf nope")


    def test_toolbox_run_shell_rejects_shell_metacharacters(self):
        from filament.agent_tools import Toolbox

        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig(use_shell=True, command_allowlist=["echo"])
            tools = Toolbox(d, cfg)
            with self.assertRaises(ValueError):
                tools.run_shell("echo ok; rm -rf /")
            with self.assertRaises(ValueError):
                tools.run_shell("echo ok | cat")

    def test_ensure_target_rejects_null_byte(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                ensure_target_within_root("safe" + chr(0) + ".py", root)

    def test_toolbox_rejects_null_byte_path(self):
        from filament.agent_tools import Toolbox

        with tempfile.TemporaryDirectory() as d:
            tools = Toolbox(d, HarnessConfig())
            with self.assertRaises(ValueError):
                tools._safe("evil" + chr(0) + ".py")

    def test_toolbox_write_file_atomic(self):
        from filament.agent_tools import Toolbox
        from filament.config import HarnessConfig

        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig(atomic_writes=True)
            tools = Toolbox(d, cfg)
            result = tools.write_file("atomic.txt", "hello", append=False)
            self.assertIn("wrote 5 chars", result)
            self.assertEqual((Path(d) / "atomic.txt").read_text(encoding="utf-8"), "hello")
            tools.write_file("atomic.txt", " world", append=True)
            self.assertEqual((Path(d) / "atomic.txt").read_text(encoding="utf-8"), "hello world\n")

    def test_ollama_client_failure_raises_runtimeerror(self):
        from unittest.mock import patch

        client = OllamaClient(model="m", base_url="http://127.0.0.1:11434")
        with patch("filament.llm.urllib.request.urlopen", side_effect=ConnectionError("boom")):
            with self.assertRaises(RuntimeError):
                client.generate("hello")


    def test_one_line_replacement_ignores_unextractable_fence(self):
        from filament.config import HarnessConfig
        from filament.orchestrator import Orchestrator
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            orch = Orchestrator(client_kind="mock", config=HarnessConfig(), shell_cwd=d)
            self.assertEqual(orch._one_line_replacement("```\n\n```"), "")

    def test_ensure_target_rejects_control_characters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                ensure_target_within_root("safe\npath.py", root)

    def test_append_code_atomic_roundtrip(self):
        from filament.files import append_code
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.py"
            path.write_text("old\n", encoding="utf-8")
            append_code(path, "new\n", atomic=True)
            self.assertEqual(path.read_text(encoding="utf-8"), "old\nnew\n")

if __name__ == "__main__":
    unittest.main()
