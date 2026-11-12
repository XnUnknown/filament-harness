"""QA regression tests for the agent loop."""
import tempfile
import unittest
from pathlib import Path

from filament.agent_loop import AgentLoop
from filament.chat import ChatOrchestrator
from filament.config import HarnessConfig
from filament.llm import ChatResult, MockLLM, ToolCall
from filament.session import create_session


class ScriptedClient:
    """Small deterministic client; unlike MockLLM it counts every model call."""

    supports_tools = True
    stream = False

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate(self, prompt, system=None):
        return "summary"

    def chat(self, messages, tools=None, on_delta=None, cancel=None):
        self.calls += 1
        item = self.script.pop(0) if self.script else {"content": "fallback"}
        calls = []
        for index, call in enumerate(item.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            calls.append(
                ToolCall(
                    id=str(call.get("id") or f"call_{index + 1}"),
                    name=str(call.get("name") or ""),
                    arguments=dict(call.get("arguments") or {}),
                )
            )
        return ChatResult(content=str(item.get("content") or ""), tool_calls=calls)


class TestStepBudgetIsPerTurn(unittest.TestCase):
    def test_second_turn_still_calls_the_model_after_max_steps_was_reached(self):
        with tempfile.TemporaryDirectory() as directory:
            config = HarnessConfig(max_steps=2, compact_with_llm=False)
            client = ScriptedClient(
                [
                    {"tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}]},
                    {"tool_calls": [{"name": "finish", "arguments": {"summary": "first done"}}]},
                    {"content": "second answer"},
                ]
            )
            chat = ChatOrchestrator(config, directory, client=client)

            first = chat.send("first request")
            self.assertEqual(first["status"], "done")
            self.assertEqual(first["steps"], 2)
            self.assertEqual(client.calls, 2)

            second = chat.send("second request")
            self.assertEqual(second["status"], "done", second)
            self.assertEqual(second["content"], "second answer")
            self.assertEqual(second["steps"], 1)
            self.assertEqual(client.calls, 3)

            # Session telemetry still accumulates steps across turns.
            self.assertEqual(chat.session.usage["steps"], 3)


class TestOrphanToolResults(unittest.TestCase):
    """A tool message without its assistant call is rejected by providers."""

    def run_with_messages(self, messages):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        session = create_session(root)
        session.messages = list(messages)
        client = MockLLM([{"content": "ok"}])
        loop = AgentLoop(HarnessConfig(compact_with_llm=False), root, session, client=client)
        result = loop.run("continue")
        self.assertEqual(result.status, "done", result)
        return loop, client.requests[0]

    def test_leading_orphan_tool_message_is_dropped(self):
        loop, sent = self.run_with_messages(
            [
                {"role": "user", "content": "start"},
                {"role": "tool", "tool_call_id": "ghost", "name": "read_file", "content": "stale"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        self.assertEqual(
            [message["role"] for message in sent],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual([m for m in sent if m.get("role") == "tool"], [])
        self.assertEqual([m for m in loop.messages if m.get("role") == "tool"], [])

    def test_foreign_result_inside_a_valid_run_is_dropped(self):
        loop, sent = self.run_with_messages(
            [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "a", "name": "read_file", "arguments": {"path": "x"}}],
                },
                {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": "good"},
                {"role": "tool", "tool_call_id": "ghost", "name": "read_file", "content": "stale"},
                {"role": "assistant", "content": "done"},
            ]
        )
        tool_ids = [m.get("tool_call_id") for m in sent if m.get("role") == "tool"]
        self.assertEqual(tool_ids, ["a"])
        self.assertEqual([m.get("tool_call_id") for m in loop.messages if m.get("role") == "tool"], ["a"])

    def test_duplicate_result_for_one_call_is_dropped(self):
        _loop, sent = self.run_with_messages(
            [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "a", "name": "read_file", "arguments": {"path": "x"}}],
                },
                {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": "good"},
                {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": "again"},
                {"role": "assistant", "content": "done"},
            ]
        )
        tool_ids = [m.get("tool_call_id") for m in sent if m.get("role") == "tool"]
        self.assertEqual(tool_ids, ["a"])
        tool_messages = [m for m in sent if m.get("role") == "tool"]
        self.assertEqual(tool_messages[0]["content"], "good")


if __name__ == "__main__":
    unittest.main()
