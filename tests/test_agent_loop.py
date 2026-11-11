import json
import tempfile
import threading
import unittest
from pathlib import Path

from filament.agent_loop import AgentLoop
from filament.chat import ChatOrchestrator
from filament.config import HarnessConfig
from filament.llm import ChatResult, MockLLM, ToolCall
from filament.session import create_session, list_sessions, load_session


def scripted(*items):
    return MockLLM(script=list(items))


class RoutingClient:
    """Parent and subagent share one client but pop from separate scripts."""

    supports_tools = True
    stream = False

    def __init__(self, parent, child):
        self.parent = list(parent)
        self.child = list(child)
        self.tool_lists = []

    def generate(self, prompt, system=None):
        return "summary"

    def chat(self, messages, tools=None, on_delta=None, cancel=None):
        system = messages[0].get("content", "") if messages else ""
        self.tool_lists.append(
            ("child" if "Subagent rules" in system else "parent", [t["name"] for t in (tools or [])])
        )
        item = (self.child if "Subagent rules" in system else self.parent).pop(0)
        calls = [
            ToolCall(
                id=str(call.get("id") or f"call_{index + 1}"),
                name=str(call["name"]),
                arguments=dict(call.get("arguments") or {}),
            )
            for index, call in enumerate(item.get("tool_calls") or [])
        ]
        return ChatResult(content=str(item.get("content") or ""), tool_calls=calls)


class AgentLoopTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_chat(self, client, config=None, mode="build", events=None):
        chat = ChatOrchestrator(
            config or HarnessConfig(),
            str(self.root),
            mode=mode,
            client=client,
            event_sink=(events.append if events is not None else None),
        )
        return chat


class TestBasicLoop(AgentLoopTestCase):
    def test_tool_call_then_finish(self):
        events = []
        client = scripted(
            {"content": "Writing it.", "tool_calls": [{"name": "write_file", "arguments": {"path": "hello.py", "content": "print('hi')\n"}}]},
            {"tool_calls": [{"name": "finish", "arguments": {"summary": "created hello.py"}}]},
        )
        chat = self.make_chat(client, events=events)
        result = chat.send("create hello.py")
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["content"], "created hello.py")
        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), "print('hi')\n")
        kinds = [event["type"] for event in events]
        self.assertIn("tool_start", kinds)
        self.assertIn("tool_end", kinds)
        self.assertIn("done", kinds)
        self.assertLess(kinds.index("tool_start"), kinds.index("done"))

    def test_session_is_persisted_with_messages(self):
        client = scripted({"tool_calls": [{"name": "finish", "arguments": {"summary": "ok"}}]})
        chat = self.make_chat(client)
        chat.send("do a thing")
        sessions = list_sessions(self.root)
        self.assertEqual(len(sessions), 1)
        loaded = load_session(sessions[0].session_id, self.root)
        self.assertEqual(loaded.messages[-1]["role"], "tool")
        self.assertEqual(loaded.title, "do a thing")

    def test_json_fallback_action_is_executed(self):
        client = scripted(
            {"content": '{"tool": "write_file", "arguments": {"path": "fallback.txt", "content": "ok"}}'},
            {"tool_calls": [{"name": "finish", "arguments": {"summary": "fallback worked"}}]},
        )
        chat = self.make_chat(client)
        result = chat.send("use the fallback")
        self.assertEqual(result["status"], "done")
        self.assertTrue((self.root / "fallback.txt").exists())

    def test_json_without_arguments_is_not_a_tool_call(self):
        client = scripted({"content": '{"name": "package", "version": "1.0"}'} )
        chat = self.make_chat(client)
        result = chat.send("just answer with json")
        self.assertEqual(result["status"], "done")
        self.assertIn('"package"', result["content"])
        tool_messages = [m for m in chat.messages if m.get("role") == "tool"]
        self.assertEqual(tool_messages, [])

    def test_max_steps_stops_the_loop(self):
        config = HarnessConfig(max_steps=1)
        client = scripted(
            {"tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}]},
            {"tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}]},
        )
        chat = self.make_chat(client, config=config)
        result = chat.send("loop forever")
        self.assertEqual(result["status"], "max_steps")
        self.assertEqual(result["steps"], 1)

    def test_repeated_call_guard_leaves_no_orphan_tool_calls(self):
        call = {"tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}]}
        client = scripted(call, dict(call), dict(call), dict(call))
        chat = self.make_chat(client)
        result = chat.send("repeat")
        self.assertEqual(result["status"], "stalled")
        pending = set()
        for message in chat.messages:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    pending.add(tool_call["id"])
            elif message.get("role") == "tool":
                pending.discard(message.get("tool_call_id"))
        self.assertEqual(pending, set())

    def test_cancel_event_stops_before_the_first_model_call(self):
        client = MockLLM()
        session = create_session(self.root)
        loop = AgentLoop(HarnessConfig(), self.root, session, client=client)
        loop._cancel.set()
        result = loop.run("stop")
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(client.requests, [])


class TestQuestions(AgentLoopTestCase):
    def test_ask_user_pauses_and_resumes(self):
        client = scripted(
            {"content": "Need a choice.", "tool_calls": [{"name": "ask_user", "arguments": {"question": "Language?", "options": ["python", "go"]}}]},
            {"tool_calls": [{"name": "write_file", "arguments": {"path": "app.py", "content": "print(1)\n"}}]},
            {"tool_calls": [{"name": "finish", "arguments": {"summary": "wrote app.py"}}]},
        )
        chat = self.make_chat(client)
        first = chat.send("build an app")
        self.assertEqual(first["status"], "awaiting_input")
        question = first.get("question") or {}
        self.assertEqual(question.get("question"), "Language?")
        self.assertEqual(chat.session.pending_question["tool_call_id"], "call_1")
        second = chat.answer(question.get("id") or "", "python")
        self.assertEqual(second["status"], "done")
        self.assertTrue((self.root / "app.py").exists())
        tool_messages = [m for m in chat.messages if m.get("role") == "tool"]
        self.assertEqual(tool_messages[0]["content"], "python")

    def test_send_is_rejected_while_a_question_is_pending(self):
        client = scripted(
            {"tool_calls": [{"name": "ask_user", "arguments": {"question": "Well?"}}]},
        )
        chat = self.make_chat(client)
        chat.send("ask me")
        with self.assertRaises(ValueError):
            chat.send("another message")


class TestModes(AgentLoopTestCase):
    def test_plan_mode_blocks_writes(self):
        events = []
        client = scripted(
            {"tool_calls": [{"name": "write_file", "arguments": {"path": "blocked.py", "content": "x"}}]},
            {"tool_calls": [{"name": "finish", "arguments": {"summary": "plan ready"}}]},
        )
        chat = self.make_chat(client, mode="plan", events=events)
        result = chat.send("write a file")
        self.assertEqual(result["status"], "done")
        self.assertFalse((self.root / "blocked.py").exists())
        failed = [e for e in events if e["type"] == "tool_end" and e.get("ok") is False]
        self.assertTrue(failed)
        self.assertIn("not available in plan mode", failed[0]["content"])


class TestSubagents(AgentLoopTestCase):
    def test_subagent_gets_own_session_and_result_returns_to_parent(self):
        events = []
        client = RoutingClient(
            parent=[
                {"tool_calls": [{"name": "spawn_agent", "arguments": {"task": "write lib.py"}}]},
                {"tool_calls": [{"name": "finish", "arguments": {"summary": "delegated"}}]},
            ],
            child=[
                {"tool_calls": [{"name": "write_file", "arguments": {"path": "lib.py", "content": "def double(x):\n    return x * 2\n"}}]},
                {"tool_calls": [{"name": "finish", "arguments": {"summary": "wrote lib.py"}}]},
            ],
        )
        chat = self.make_chat(client, events=events)
        result = chat.send("delegate lib.py")
        self.assertEqual(result["status"], "done")
        self.assertIn("lib.py", result["files_changed"])
        self.assertTrue((self.root / "lib.py").exists())

        sessions = list_sessions(self.root, kind="subagent")
        self.assertEqual(len(sessions), 1)
        child = sessions[0]
        self.assertEqual(child.kind, "subagent")
        self.assertEqual(child.parent_id, chat.session.session_id)
        self.assertEqual(child.status, "done")

        starts = [e for e in events if e["type"] == "subagent_start"]
        ends = [e for e in events if e["type"] == "subagent_end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["child_session_id"], child.session_id)

        child_tools = [names for kind, names in client.tool_lists if kind == "child"]
        self.assertTrue(child_tools)
        self.assertNotIn("spawn_agent", child_tools[0])
        self.assertNotIn("ask_user", child_tools[0])

    def test_depth_limit_prevents_nested_spawn(self):
        client = RoutingClient(
            parent=[
                {"tool_calls": [{"name": "spawn_agent", "arguments": {"task": "child"}}]},
                {"tool_calls": [{"name": "finish", "arguments": {"summary": "done"}}]},
            ],
            child=[{"tool_calls": [{"name": "finish", "arguments": {"summary": "child done"}}]}],
        )
        chat = self.make_chat(client)
        chat.send("delegate")
        child_tools = [names for kind, names in client.tool_lists if kind == "child"]
        self.assertNotIn("spawn_agent", child_tools[0])


if __name__ == "__main__":
    unittest.main()
