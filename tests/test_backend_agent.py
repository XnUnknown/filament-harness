import io
import json
import tempfile
import unittest
from pathlib import Path

from filament import backend_server
from filament.chat import ChatOrchestrator
from filament.config import HarnessConfig
from filament.llm import MockLLM


class BackendAgentTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.client = MockLLM(
            script=[
                {
                    "content": "working",
                    "tool_calls": [
                        {"name": "write_file", "arguments": {"path": "a.txt", "content": "hello"}}
                    ],
                },
                {"tool_calls": [{"name": "finish", "arguments": {"summary": "wrote a.txt"}}]},
            ]
        )
        self.chat = ChatOrchestrator(
            self._config(), str(self.root), client=self.client
        )
        backend_server.STATE.update(
            {"workspace": self.root, "config": self.chat.config, "chat": self.chat, "busy": False}
        )
        self.buffer = io.BytesIO()
        backend_server.EVENT_STREAM = self.buffer

    def _config(self):
        config = HarnessConfig()
        config.llm.kind = "mock"
        return config

    def tearDown(self):
        backend_server.EVENT_STREAM = None
        backend_server.STATE.update(
            {"workspace": None, "config": None, "chat": None, "busy": False}
        )
        self.tmp.cleanup()

    def call(self, method, params=None, request_id=1):
        payload = {"id": request_id, "method": method, "params": params or {}}
        return json.loads(backend_server.respond(json.dumps(payload)))

    def events(self):
        raw = self.buffer.getvalue().decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


class TestBackendChat(BackendAgentTestCase):
    def test_chat_streams_events_and_returns_result(self):
        reply = self.call("chat", {"message": "make a.txt"})
        self.assertTrue(reply["ok"])
        result = reply["result"]
        self.assertEqual(result["type"], "done")
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["content"], "wrote a.txt")
        self.assertIn("a.txt", result["files_changed"])
        self.assertTrue((self.root / "a.txt").exists())

        kinds = [event["event"] for event in self.events()]
        for expected in ("session", "delta", "tool_start", "tool_end", "done"):
            self.assertIn(expected, kinds)
        self.assertTrue(all("event" in event for event in self.events()))

    def test_question_and_answer_round_trip(self):
        self.chat.client = MockLLM(
            script=[
                {
                    "tool_calls": [
                        {"name": "ask_user", "arguments": {"question": "Which name?", "options": ["a", "b"]}}
                    ]
                },
                {"tool_calls": [{"name": "finish", "arguments": {"summary": "named it b"}}]},
            ]
        )
        first = self.call("chat", {"message": "name it"})
        self.assertTrue(first["ok"])
        result = first["result"]
        self.assertEqual(result["type"], "question")
        self.assertEqual(result["question"]["question"], "Which name?")
        self.assertIn("question", [event["event"] for event in self.events()])

        answer = self.call(
            "answer", {"question_id": result["question"]["id"], "text": "b"}, request_id=2
        )
        self.assertTrue(answer["ok"], answer)
        self.assertEqual(answer["result"]["status"], "done")

    def test_second_chat_is_rejected_while_busy(self):
        backend_server.STATE["busy"] = True
        reply = self.call("chat", {"message": "another"})
        self.assertFalse(reply["ok"])
        self.assertIn("already running", reply["error"])
        backend_server.STATE["busy"] = False

    def test_session_switch_is_rejected_while_busy(self):
        self.chat.save()
        session_file = (
            self.root / ".filament" / "sessions" / f"{self.chat.session.session_id}.json"
        )
        self.assertTrue(session_file.exists())
        backend_server.STATE["busy"] = True
        try:
            for method, params in (
                ("load_session", {"session_id": self.chat.session.session_id}),
                ("new_session", {}),
                ("select_workspace", {"path": str(self.root)}),
                ("delete_session", {"session_id": self.chat.session.session_id}),
                ("compress_context", {}),
            ):
                reply = self.call(method, params)
                self.assertFalse(reply["ok"], method)
                self.assertIn("running", reply["error"], method)
        finally:
            backend_server.STATE["busy"] = False
        # The rejected switch must not have replaced the live session.
        self.assertIs(backend_server.STATE["chat"], self.chat)
        # Nor may the rejected delete have removed its file.
        self.assertTrue(session_file.exists())

    def test_cancel_marks_the_session(self):
        reply = self.call("cancel")
        self.assertTrue(reply["ok"])
        self.assertTrue(self.chat._cancel.is_set())


class TestBackendSessions(BackendAgentTestCase):
    def test_new_session_set_mode_list_and_load(self):
        chat_reply = self.call("chat", {"message": "first"})
        session_id = chat_reply["result"]["session_id"]

        new = self.call("new_session", {"mode": "plan"}, request_id=2)
        self.assertTrue(new["ok"])
        self.assertEqual(new["result"]["session"]["mode"], "plan")
        self.assertNotEqual(new["result"]["session"]["session_id"], session_id)

        mode = self.call("set_mode", {"mode": "build"}, request_id=3)
        self.assertEqual(mode["result"]["mode"], "build")

        sessions = self.call("list_sessions", {}, request_id=4)
        ids = [item["session_id"] for item in sessions["result"]]
        self.assertIn(session_id, ids)

        loaded = self.call("load_session", {"session_id": session_id}, request_id=5)
        self.assertTrue(loaded["ok"])
        self.assertEqual(loaded["result"]["session"]["session_id"], session_id)
        self.assertTrue(loaded["result"]["messages"])

    def test_get_state_includes_provider_and_context(self):
        reply = self.call("get_state")
        self.assertTrue(reply["ok"])
        state = reply["result"]["state"]
        self.assertEqual(state["provider"]["kind"], "mock")
        self.assertGreater(state["limit"], 0)
        self.assertIn("messages", state)

    def test_read_file_and_list_dir(self):
        (self.root / "x.py").write_text("print(1)\n", encoding="utf-8")
        read = self.call("read_file", {"path": "x.py"})
        self.assertIn("print(1)", read["result"]["content"])
        listing = self.call("list_dir", {"path": "."})
        names = [entry["name"] for entry in listing["result"]["entries"]]
        self.assertIn("x.py", names)

    def test_delete_session(self):
        chat_reply = self.call("chat", {"message": "first"})
        session_id = chat_reply["result"]["session_id"]
        deleted = self.call("delete_session", {"session_id": session_id}, request_id=2)
        self.assertTrue(deleted["result"]["deleted"])
        sessions = self.call("list_sessions", {}, request_id=3)
        self.assertNotIn(session_id, [item["session_id"] for item in sessions["result"]])

    def test_deleting_the_open_session_replaces_it(self):
        chat_reply = self.call("chat", {"message": "first"})
        old_id = chat_reply["result"]["session_id"]
        deleted = self.call("delete_session", {"session_id": old_id}, request_id=2)
        result = deleted["result"]
        self.assertTrue(result["deleted"])
        # The next turn saves the open session again, so a deleted session that
        # is still loaded would reappear on disk. It must be closed instead.
        new_id = result["session"]["session_id"]
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(backend_server.STATE["chat"].session.session_id, new_id)
        sessions = self.call("list_sessions", {}, request_id=3)
        ids = [item["session_id"] for item in sessions["result"]]
        self.assertNotIn(old_id, ids)
        self.assertIn(new_id, ids)


if __name__ == "__main__":
    unittest.main()
