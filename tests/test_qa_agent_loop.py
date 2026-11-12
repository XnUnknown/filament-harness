"""QA regression tests for the agent loop."""
import tempfile
import unittest
from pathlib import Path

from filament.chat import ChatOrchestrator
from filament.config import HarnessConfig
from filament.llm import ChatResult, ToolCall


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


if __name__ == "__main__":
    unittest.main()
