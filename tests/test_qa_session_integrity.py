"""Junk values in a session file must not brick the session or the UI.

Session files are declarative data: they can be hand-edited, copied, written by
an older build, or truncated by a crash.  ``Session.__post_init__`` promises to
normalize the containers so one bad field cannot take the whole session down.
These tests cover the two shapes that used to escape that normalization: a
non-list ``tool_calls`` value (which crashed token estimation before the first
model call) and a non-list ``pending_question.options`` (which crashed the
renderer while rebuilding a question card).
"""
import json
import tempfile
import unittest
from pathlib import Path

from filament.agent_loop import AgentLoop
from filament.chat import ChatOrchestrator
from filament.compaction import estimate_tokens
from filament.config import HarnessConfig
from filament.llm import ChatResult, ToolCall, _anthropic_messages, _openai_message
from filament.session import create_session, load_session, save_session, session_dir


class AdapterFidelityClient:
    """Convert messages exactly as the provider adapters do, then answer.

    A plain MockLLM never touches ``tool_calls``; the real adapters iterate it.
    Using the real converters here means a malformed history fails the test the
    same way it would fail against a live provider.
    """

    supports_tools = True
    stream = False

    def __init__(self):
        self.requests = []

    def generate(self, prompt, system=None):
        return "summary"

    def chat(self, messages, tools=None, on_delta=None, cancel=None):
        self.requests.append(list(messages))
        for message in messages:
            _openai_message(message)
        _anthropic_messages(messages)
        return ChatResult(content="ok")


def write_session_file(root: Path, session_id: str, messages, **extra) -> None:
    """Write a session JSON directly, bypassing Session's own cleaning."""
    directory = session_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "cwd": str(root),
        "mode": "build",
        "kind": "main",
        "title": "junk",
        "status": "idle",
        "messages": messages,
    }
    payload.update(extra)
    (directory / f"{session_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class TestJunkToolCalls(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_token_estimation_tolerates_junk_shapes(self):
        # Before the fix the scalar was iterated and raised TypeError.
        self.assertGreater(
            estimate_tokens(
                [{"role": "assistant", "content": "hi", "tool_calls": 5}]
            ),
            0,
        )
        self.assertEqual(estimate_tokens(["not a message"]), 1)  # type: ignore[list-item]

    def test_loop_survives_a_scalar_tool_calls(self):
        session = create_session(self.root)
        session.messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "hi", "tool_calls": 5},
        ]
        client = AdapterFidelityClient()
        loop = AgentLoop(
            HarnessConfig(compact_with_llm=False), self.root, session, client=client
        )

        result = loop.run("continue")

        self.assertEqual(result.status, "done", result)
        assistant = next(
            message for message in loop.messages if message.get("role") == "assistant"
        )
        self.assertNotIn("tool_calls", assistant)
        self.assertTrue(client.requests)

    def test_loading_from_disk_drops_a_scalar_tool_calls(self):
        write_session_file(
            self.root,
            "20260101-000000-junk",
            [
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": "hi", "tool_calls": 5},
            ],
        )

        loaded = load_session("20260101-000000-junk", self.root)

        self.assertNotIn("tool_calls", loaded.messages[1])
        # The loaded transcript must be safe to estimate (loop construction
        # and ChatOrchestrator.state() both do this before any model call).
        self.assertGreater(estimate_tokens(loaded.messages), 0)

    def test_loading_from_disk_drops_non_object_call_entries(self):
        write_session_file(
            self.root,
            "20260101-000000-calls",
            [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "a", "name": "read_file", "arguments": {"path": "x"}},
                        "junk",
                        7,
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": "x"},
            ],
        )

        loaded = load_session("20260101-000000-calls", self.root)

        self.assertEqual(
            [call["id"] for call in loaded.messages[1]["tool_calls"]], ["a"]
        )

    def test_saved_and_reloaded_session_round_trips_cleanly(self):
        session = create_session(self.root)
        session.messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "hi", "tool_calls": 5},
        ]
        save_session(session, self.root)

        loaded = load_session(session.session_id, self.root)

        self.assertNotIn("tool_calls", loaded.messages[1])


class TestJunkPendingQuestion(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, **pending):
        write_session_file(
            self.root,
            "20260101-000000-question",
            [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "name": "ask_user", "arguments": {"question": "?"}}
                    ],
                },
            ],
            status="awaiting_input",
            pending_question={
                "id": "q1",
                "tool_call_id": "c1",
                "question": "Pick one",
                "options": ["yes"],
                **pending,
            },
        )
        return load_session("20260101-000000-question", self.root)

    def test_non_list_options_become_an_empty_list(self):
        session = self._load(options="yes")

        self.assertIsNotNone(session.pending_question)
        self.assertEqual(session.pending_question["options"], [])
        self.assertEqual(session.status, "awaiting_input")
        # The renderer iterates this value; it must be JSON-serializable too.
        json.dumps(session.pending_question)

    def test_non_string_options_are_coerced_and_blank_entries_dropped(self):
        session = self._load(options=[1, "two", "", "  ", None])

        self.assertIsNotNone(session.pending_question)
        self.assertEqual(session.pending_question["options"], ["1", "two"])

    def test_non_string_question_becomes_text(self):
        session = self._load(question=5)

        self.assertIsNotNone(session.pending_question)
        self.assertEqual(session.pending_question["question"], "5")


class _AskingClient:
    supports_tools = True
    stream = False

    def generate(self, prompt, system=None):
        return "summary"

    def chat(self, messages, tools=None, on_delta=None, cancel=None):
        return ChatResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="ask-1",
                    name="ask_user",
                    arguments={"question": "Pick one", "options": ["red", "blue"]},
                )
            ],
        )


class _DoneClient:
    supports_tools = True
    stream = False

    def generate(self, prompt, system=None):
        return "summary"

    def chat(self, messages, tools=None, on_delta=None, cancel=None):
        return ChatResult(content="done")


class TestValidQuestionRoundTrip(unittest.TestCase):
    """Normalization must not damage a question the harness wrote itself."""

    def test_saved_question_keeps_its_fields_and_is_answerable_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            config = HarnessConfig(compact_with_llm=False)
            chat = ChatOrchestrator(config, directory, client=_AskingClient())
            first = chat.send("start")
            self.assertEqual(first["status"], "awaiting_input")
            question_id = first["question"]["id"]

            reloaded = ChatOrchestrator(config, directory, client=_DoneClient())
            reloaded.load_session(chat.session.session_id)
            pending = reloaded.session.pending_question

            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertEqual(pending["id"], question_id)
            self.assertEqual(pending["question"], "Pick one")
            self.assertEqual(pending["options"], ["red", "blue"])
            self.assertEqual(pending["tool_call_id"], "ask-1")
            self.assertEqual(reloaded.session.status, "awaiting_input")

            result = reloaded.answer(question_id, "red")
            self.assertEqual(result["status"], "done", result)


if __name__ == "__main__":
    unittest.main()
