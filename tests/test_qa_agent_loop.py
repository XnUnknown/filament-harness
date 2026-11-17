"""QA regression tests for the agent loop."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from filament.agent_loop import AgentLoop
from filament.chat import ChatOrchestrator
from filament.config import HarnessConfig
from filament.llm import ChatResult, MockLLM, ToolCall, _anthropic_messages, _openai_message
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


class TestNamelessToolCallInSession(unittest.TestCase):
    """A call the adapters drop must not leave an orphan result on the wire."""

    def test_nameless_call_and_its_result_never_reach_the_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = create_session(root)
            session.messages = [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "x1", "arguments": {}}],
                },
                {"role": "tool", "tool_call_id": "x1", "name": "", "content": "did something"},
            ]
            client = MockLLM([{"content": "ok"}])
            loop = AgentLoop(HarnessConfig(compact_with_llm=False), root, session, client=client)

            result = loop.run("continue")

            self.assertEqual(result.status, "done", result)
            wire = [_openai_message(message) for message in client.requests[0]]
            issued = {
                str(call.get("id"))
                for message in wire
                if message.get("role") == "assistant"
                for call in (message.get("tool_calls") or [])
            }
            tool_ids = [m["tool_call_id"] for m in wire if m.get("role") == "tool"]
            self.assertEqual([tool_id for tool_id in tool_ids if tool_id not in issued], [])

            _system, conversation = _anthropic_messages(loop.messages)
            tool_use_ids = {
                block["id"]
                for turn in conversation
                if turn["role"] == "assistant"
                for block in turn["content"]
                if block.get("type") == "tool_use"
            }
            result_ids = {
                block["tool_use_id"]
                for turn in conversation
                for block in turn["content"]
                if block.get("type") == "tool_result"
            }
            self.assertEqual(result_ids - tool_use_ids, set())


class TestFallbackFinalReply(unittest.TestCase):
    """The prompt's ``{"final": ...}`` envelope must not reach the user raw."""

    def reply(self, content):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = MockLLM([{"content": content}])
            loop = AgentLoop(
                HarnessConfig(compact_with_llm=False), root, create_session(root), client=client
            )
            return loop.run("answer me")

    def test_final_json_is_unwrapped(self):
        result = self.reply('{"final": "The answer is forty-two."}')
        self.assertEqual(result.status, "done")
        self.assertEqual(result.content, "The answer is forty-two.")

    def test_fenced_final_json_is_unwrapped(self):
        result = self.reply('```json\n{"final": "Fenced answer"}\n```')
        self.assertEqual(result.content, "Fenced answer")

    def test_plain_json_answer_is_still_reported_verbatim(self):
        raw = '{"name": "package", "version": "1.0"}'
        result = self.reply(raw)
        self.assertEqual(result.content, raw)


class TestCallIdsAreUnique(unittest.TestCase):
    """Duplicate or missing call ids must be replaced before the turn is saved.

    Results are paired by id; the renderer's history rebuild used a global id
    map, so a reused id paired an old card with a newer result.
    """

    class FixedIdClient:
        supports_tools = True
        stream = False

        def __init__(self, ids):
            self.ids = list(ids)
            self.turns = 0

        def generate(self, prompt, system=None):
            return "summary"

        def chat(self, messages, tools=None, on_delta=None, cancel=None):
            self.turns += 1
            if self.turns == 1:
                calls = [
                    ToolCall(id=value, name="list_dir", arguments={"path": "."})
                    for value in self.ids
                ]
                return ChatResult(content="", tool_calls=calls)
            return ChatResult(content="done")

    def test_missing_and_duplicate_ids_are_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop = AgentLoop(
                HarnessConfig(compact_with_llm=False),
                root,
                create_session(root),
                client=self.FixedIdClient(["", "", "dup", "dup"]),
            )
            result = loop.run("go")
            issued = [
                call["id"]
                for message in loop.messages
                if message.get("role") == "assistant"
                for call in message.get("tool_calls") or []
            ]
            results = [
                message["tool_call_id"]
                for message in loop.messages
                if message.get("role") == "tool"
            ]
        self.assertEqual(result.status, "done", result)
        self.assertEqual(len(issued), 4)
        self.assertEqual(len(set(issued)), 4)
        self.assertEqual(results, issued)


class TestFailedAnswerKeepsTheQuestion(unittest.TestCase):
    """A rejected answer must leave the question answerable.

    The Electron renderer restores the card after a failed `answer` request;
    that is only correct while the session still holds the question.
    """

    def test_mismatched_answer_does_not_clear_the_pending_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = ScriptedClient(
                [
                    {"tool_calls": [{"name": "ask_user", "arguments": {"question": "Which?"}}]},
                    {"content": "done"},
                ]
            )
            chat = ChatOrchestrator(
                HarnessConfig(compact_with_llm=False), str(root), client=client
            )
            first = chat.send("ask me")
            self.assertEqual(first["status"], "awaiting_input")
            question_id = (first.get("question") or {}).get("id") or ""

            with self.assertRaises(ValueError):
                chat.answer("wrong-id", "blue")
            self.assertIsNotNone(chat.session.pending_question)

            second = chat.answer(question_id, "blue")
            self.assertEqual(second["status"], "done")
            self.assertIsNone(chat.session.pending_question)


class TestQuestionIdsAreUnique(unittest.TestCase):
    """Two questions in one session must never share an id.

    The Electron renderer keys question cards by ``data-question-id``.  Ids were
    built from ``int(time.time())`` plus the step counter, and the step counter
    resets every turn, so a second question asked in the same wall-clock second
    at the same step reused the id of the first, already answered card: the new
    question rendered nowhere and the composer stayed disabled.
    """

    class AskTwiceClient:
        supports_tools = True
        stream = False

        def __init__(self):
            self.turns = 0

        def generate(self, prompt, system=None):
            return "summary"

        def chat(self, messages, tools=None, on_delta=None, cancel=None):
            self.turns += 1
            if self.turns <= 2:
                return ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{self.turns}",
                            name="ask_user",
                            arguments={"question": f"question {self.turns}"},
                        )
                    ],
                )
            return ChatResult(content="finished")

    def test_questions_asked_in_the_same_second_get_distinct_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = ChatOrchestrator(
                HarnessConfig(compact_with_llm=False),
                directory,
                client=self.AskTwiceClient(),
            )
            # Pin the clock the old id generator used; the ids must not collide
            # even when both questions are asked in the same second.
            with patch("filament.agent_loop.time.time", return_value=1_700_000_000.0):
                first = chat.send("ask me")
                first_id = (first.get("question") or {}).get("id") or ""
                second = chat.answer(first_id, "one")
                second_id = (second.get("question") or {}).get("id") or ""
                done = chat.answer(second_id, "two")

        self.assertTrue(first_id)
        self.assertTrue(second_id)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(done["status"], "done")


class TestInterruptedTurnKeepsTranscript(unittest.TestCase):
    """Ctrl+C during a model call must not let the next save erase the turn."""

    class InterruptingClient:
        supports_tools = True
        stream = False

        def __init__(self):
            self.calls = 0

        def generate(self, prompt, system=None):
            return "summary"

        def chat(self, messages, tools=None, on_delta=None, cancel=None):
            self.calls += 1
            raise KeyboardInterrupt

    def test_keyboard_interrupt_keeps_the_user_message_in_the_orchestrator(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = ChatOrchestrator(
                HarnessConfig(compact_with_llm=False),
                directory,
                client=self.InterruptingClient(),
            )
            with self.assertRaises(KeyboardInterrupt):
                chat.send("write the report")
            self.assertEqual(len(chat.messages), 2)  # system + user
            self.assertEqual(chat.messages[-1]["role"], "user")
            self.assertEqual(chat.messages[-1]["content"], "write the report")
            chat.save()
            reloaded = ChatOrchestrator(
                HarnessConfig(compact_with_llm=False),
                directory,
                client=self.InterruptingClient(),
            )
            reloaded.load_session(chat.session.session_id)
            self.assertEqual(reloaded.messages[-1]["content"], "write the report")


if __name__ == "__main__":
    unittest.main()
