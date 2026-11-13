import json
import tempfile
import threading
import unittest
from pathlib import Path

from filament.agent_loop import AgentLoop
from filament.chat import ChatOrchestrator
from filament.config import HarnessConfig
from filament.llm import ChatResult, MockLLM, ToolCall
from filament.session import Session, create_session, list_sessions, load_session, save_session


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


class TestLoopCompaction(AgentLoopTestCase):
    def test_wide_read_batch_is_compacted_before_the_next_call(self):
        # Regression: one assistant turn with many large tool results used to
        # make the turn cut give up, so the next model call went out over
        # budget with the whole batch still attached.
        (self.root / "big.txt").write_text("x" * 30_000, encoding="utf-8")
        calls = [
            {"name": "read_file", "arguments": {"path": "big.txt"}} for _ in range(5)
        ]
        client = scripted(
            {"tool_calls": calls},
            {"tool_calls": [{"name": "finish", "arguments": {"summary": "handled"}}]},
        )
        config = HarnessConfig()
        config.llm.kind = "mock"
        config.max_tokens = 8000
        config.compact_threshold = 0.6
        config.keep_recent_messages = 6
        config.compact_with_llm = False
        config.max_depth = 0
        events = []
        chat = self.make_chat(client, config=config, events=events)
        result = chat.send("read it five times")
        self.assertEqual(result["status"], "done")
        compactions = [event for event in events if event["type"] == "compaction"]
        self.assertTrue(compactions, "expected a compaction event")
        stats = compactions[-1]
        self.assertGreater(stats["summarized_messages"], 0)
        self.assertLess(stats["after_tokens"], stats["before_tokens"])
        self.assertLessEqual(stats["after_tokens"], config.max_tokens)
        # The batch keeps its pairing and is not dropped wholesale.
        pending = set()
        kept_calls = 0
        for message in chat.messages:
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    pending.add(tool_call["id"])
                    kept_calls += 1
            elif message.get("role") == "tool":
                self.assertIn(message.get("tool_call_id"), pending)
                pending.discard(message.get("tool_call_id"))
        self.assertEqual(pending, set())
        self.assertGreaterEqual(kept_calls, 5)
        self.assertNotEqual(chat.messages[1]["role"], "tool")


class TestCancellation(AgentLoopTestCase):
    def _run_cancelling(self, result):
        holder = {}

        class CancelMidCall:
            supports_tools = True
            stream = False

            def generate(self, prompt, system=None):
                return ""

            def chat(self, messages, tools=None, on_delta=None, cancel=None):
                holder["loop"]._cancel.set()
                return result

        loop = AgentLoop(
            HarnessConfig(), self.root, create_session(self.root), client=CancelMidCall()
        )
        holder["loop"] = loop
        return loop, loop.run("stop me")

    def test_cancel_during_the_model_call_reports_cancelled(self):
        _loop, result = self._run_cancelling(ChatResult())
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.content, "Stopped at your request.")

    def test_cancel_does_not_execute_tools_from_the_partial_stream(self):
        partial = ChatResult(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "cancel.txt", "content": "x\n"},
                )
            ]
        )
        loop, result = self._run_cancelling(partial)
        self.assertEqual(result.status, "cancelled")
        self.assertFalse((self.root / "cancel.txt").exists())
        tool_messages = [message for message in loop.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        self.assertIn("cancelled", tool_messages[0]["content"])

    def test_cancel_between_tools_skips_the_rest_of_the_batch(self):
        client = scripted(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "write_file",
                        "arguments": {"path": "first.txt", "content": "one\n"},
                    },
                    {
                        "id": "call_2",
                        "name": "write_file",
                        "arguments": {"path": "second.txt", "content": "two\n"},
                    },
                ]
            },
        )
        events = []
        loop = AgentLoop(
            HarnessConfig(),
            self.root,
            create_session(self.root),
            client=client,
            event_sink=events.append,
        )

        def sink(event):
            events.append(event)
            if event.get("type") == "tool_end":
                loop.cancel()

        loop.sink = sink
        result = loop.run("write both files")

        self.assertEqual(result.status, "cancelled")
        self.assertTrue((self.root / "first.txt").exists())
        self.assertFalse((self.root / "second.txt").exists())
        results = {
            message["tool_call_id"]: message["content"]
            for message in loop.messages
            if message.get("role") == "tool"
        }
        self.assertEqual(set(results), {"call_1", "call_2"})
        self.assertIn("cancelled", results["call_2"])


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

    def test_cancel_at_the_question_prompt_unblocks_the_session(self):
        client = scripted(
            {"tool_calls": [{"name": "ask_user", "arguments": {"question": "Well?"}}]},
            {"content": "moving on"},
        )
        chat = self.make_chat(client)
        first = chat.send("ask me")
        self.assertEqual(first["status"], "awaiting_input")
        self.assertTrue(chat.cancel())
        self.assertIsNone(chat.session.pending_question)
        second = chat.send("never mind")
        self.assertEqual(second["status"], "done")
        sent = client.requests[-1]
        tool_messages = [m for m in sent if m.get("role") == "tool"]
        self.assertTrue(
            any("did not answer" in str(m.get("content")) for m in tool_messages)
        )


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


class TestToolPairingRepair(AgentLoopTestCase):
    def _session(self, messages, pending=None):
        session = create_session(self.root)
        session.messages = list(messages)
        session.pending_question = pending
        return session

    def test_interrupted_tool_call_gets_a_result_before_the_next_request(self):
        session = self._session(
            [
                {"role": "user", "content": "do a thing"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_x", "name": "run_shell", "arguments": {"command": "echo hi"}}
                    ],
                },
            ]
        )
        client = MockLLM([{"content": "resumed"}])
        loop = AgentLoop(HarnessConfig(), self.root, session, client=client)
        result = loop.run("continue please")
        self.assertEqual(result.status, "done")
        sent = client.requests[0]
        self.assertEqual(
            [message["role"] for message in sent],
            ["system", "user", "assistant", "tool", "user"],
        )
        repaired = sent[3]
        self.assertEqual(repaired["tool_call_id"], "call_x")
        self.assertTrue(repaired.get("is_error"))
        self.assertIn("interrupted", repaired["content"])

    def test_unanswered_question_is_closed_when_a_new_message_starts(self):
        session = self._session(
            [
                {"role": "user", "content": "build it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_q", "name": "ask_user", "arguments": {"question": "Which?"}}
                    ],
                },
            ],
            pending={
                "id": "q_1",
                "tool_call_id": "call_q",
                "question": "Which?",
                "options": [],
            },
        )
        client = MockLLM([{"content": "moving on"}])
        loop = AgentLoop(HarnessConfig(), self.root, session, client=client)
        loop.run("do something else")
        sent = client.requests[0]
        tool_messages = [message for message in sent if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_q")
        self.assertIn("did not answer", tool_messages[0]["content"])

    def test_repair_leaves_a_paired_conversation_unchanged(self):
        client = scripted(
            {"tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}]},
            {"content": "done"},
        )
        chat = self.make_chat(client)
        chat.send("look around")
        tool_messages = [message for message in chat.messages if message.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertNotIn("Not executed", tool_messages[0]["content"])

    def test_answer_without_its_question_call_is_sent_as_a_new_request(self):
        # A saved session can name a pending question whose assistant turn is
        # gone. Appending the answer as a tool message would be an orphan call
        # id that every provider rejects, so it must become a user message.
        session = create_session(self.root)
        session.messages = [{"role": "user", "content": "start"}]
        session.pending_question = {
            "id": "q_missing",
            "tool_call_id": "call_missing",
            "question": "Which?",
            "options": [],
        }
        client = MockLLM([{"content": "continuing"}])
        loop = AgentLoop(HarnessConfig(), self.root, session, client=client)
        result = loop.resume("blue")
        self.assertEqual(result.status, "done")
        sent = client.requests[0]
        self.assertNotIn("call_missing", [m.get("tool_call_id") for m in sent if m.get("role") == "tool"])
        self.assertEqual([m["role"] for m in sent], ["system", "user", "user"])
        self.assertEqual(sent[-1]["content"], "blue")


class TestCorruptSessionFiles(AgentLoopTestCase):
    def _save(self, **fields):
        session = create_session(self.root)
        for key, value in fields.items():
            setattr(session, key, value)
        save_session(session, self.root)
        return session

    def test_non_list_messages_do_not_break_listing_or_loading(self):
        session = self._save(messages="not a list")
        sessions = list_sessions(self.root)
        self.assertEqual([item.session_id for item in sessions], [session.session_id])
        self.assertEqual(sessions[0].display_title, session.session_id)

        loaded = load_session(session.session_id, self.root)
        self.assertEqual(loaded.messages, [])
        loop = AgentLoop(HarnessConfig(), self.root, loaded, client=MockLLM())
        self.assertEqual([message["role"] for message in loop.messages], ["system"])

    def test_junk_entries_and_wrong_container_types_are_dropped(self):
        session = self._save(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                "junk",
                7,
                None,
                {"content": "no role"},
                {"role": "unknown", "content": "x"},
                {"role": "tool", "content": "no call id"},
            ],
            files_changed="a.txt",
            todos=["not a todo", {"content": "work", "status": "pending"}],
            child_sessions={"a": 1},
            usage="nope",
        )
        # save_session refreshes updated_at, so corrupt the JSON afterwards to
        # prove a non-numeric timestamp cannot break the session list sort.
        path = self.root / ".filament" / "sessions" / f"{session.session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["updated_at"] = "yesterday"
        path.write_text(json.dumps(data), encoding="utf-8")
        sessions = list_sessions(self.root)
        self.assertEqual(len(sessions), 1)
        loaded = sessions[0]
        self.assertEqual([message["role"] for message in loaded.messages], ["user", "assistant"])
        self.assertEqual(loaded.files_changed, [])
        self.assertEqual(loaded.todos, [{"content": "work", "status": "pending"}])
        self.assertEqual(loaded.child_sessions, [])
        self.assertEqual(loaded.usage, {})
        self.assertIsInstance(loaded.updated_at, float)
        self.assertGreaterEqual(loaded.updated_at, 0)
        self.assertEqual(loaded.session_id, session.session_id)

    def test_unanswerable_pending_question_is_dropped(self):
        session = self._save(pending_question={"question": "no ids"}, status="awaiting_input")
        loaded = load_session(session.session_id, self.root)
        self.assertIsNone(loaded.pending_question)
        self.assertEqual(loaded.status, "idle")

    def test_copied_session_file_uses_its_filename_as_identity(self):
        # Copying a session file by hand must create a loadable session under
        # the new name, not a second handle to the original id: saving the
        # loaded copy would otherwise overwrite the original file and leave
        # the requested copy untouched.
        session = self._save(title="original")
        directory = self.root / ".filament" / "sessions"
        original_path = directory / f"{session.session_id}.json"
        copied_path = directory / "copied-session.json"
        copied_path.write_text(original_path.read_text(encoding="utf-8"), encoding="utf-8")

        listed = {item.session_id for item in list_sessions(self.root)}
        self.assertEqual(listed, {session.session_id, "copied-session"})

        loaded = load_session("copied-session", self.root)
        self.assertEqual(loaded.session_id, "copied-session")
        loaded.title = "copy"
        save_session(loaded, self.root)
        self.assertEqual(json.loads(original_path.read_text(encoding="utf-8"))["title"], "original")
        self.assertEqual(json.loads(copied_path.read_text(encoding="utf-8"))["title"], "copy")


class TestToolOutputClipping(AgentLoopTestCase):
    def test_provider_tool_id_cannot_escape_the_runs_directory(self):
        config = HarnessConfig(tool_output_limit=256)
        loop = AgentLoop(config, self.root, create_session(self.root), client=MockLLM())
        runs = (self.root / ".filament" / "runs").resolve()
        for call_id in ("../../escape", "..\\..\\escape", "/abs/escape", "ok-id"):
            with self.subTest(call_id=call_id):
                call = ToolCall(id=call_id, name="read_file")
                clipped = loop._clip_tool_output(call, "x" * 4000)
                self.assertIn("[full output saved:", clipped)
                saved = clipped.split("[full output saved:", 1)[1].strip().rstrip("]")
                path = (self.root / saved).resolve()
                self.assertTrue(path.is_relative_to(runs))
                self.assertTrue(path.exists())
                self.assertNotIn("..", path.name)
                self.assertFalse(any(part == ".." for part in Path(saved).parts))
                path.unlink()


class TestModelFailures(AgentLoopTestCase):
    class FlakyClient:
        supports_tools = True
        stream = True

        def __init__(self, failures: int, partial: str = ""):
            self.failures = failures
            self.partial = partial
            self.calls = 0

        def generate(self, prompt, system=None):
            return ""

        def chat(self, messages, tools=None, on_delta=None, cancel=None):
            self.calls += 1
            if on_delta and self.partial:
                on_delta(self.partial)
            if self.calls <= self.failures:
                raise RuntimeError("connection dropped")
            return ChatResult(content="recovered")

    def test_failed_model_call_closes_the_assistant_turn(self):
        events = []
        client = self.FlakyClient(failures=2)
        loop = AgentLoop(
            HarnessConfig(),
            self.root,
            create_session(self.root),
            client=client,
            event_sink=events.append,
        )
        result = loop.run("go")
        self.assertEqual(result.status, "error")
        kinds = [event["type"] for event in events]
        self.assertLess(kinds.index("assistant_start"), kinds.index("assistant_end"))
        self.assertLess(kinds.index("assistant_end"), kinds.index("error"))
        self.assertEqual(client.calls, 2)
        ends = [event for event in events if event["type"] == "assistant_end"]
        self.assertEqual(ends[-1]["content"], "")

    def test_partial_stream_is_not_retried(self):
        events = []
        client = self.FlakyClient(failures=10, partial="PART")
        loop = AgentLoop(
            HarnessConfig(),
            self.root,
            create_session(self.root),
            client=client,
            event_sink=events.append,
        )
        result = loop.run("go")
        self.assertEqual(result.status, "error")
        self.assertEqual(client.calls, 1)
        self.assertEqual(
            [event["text"] for event in events if event["type"] == "delta"], ["PART"]
        )
        ends = [event for event in events if event["type"] == "assistant_end"]
        self.assertEqual(ends[-1]["content"], "PART")
        self.assertEqual([message["role"] for message in loop.messages], ["system", "user"])
        self.assertIn("connection dropped", result.content)


if __name__ == "__main__":
    unittest.main()
