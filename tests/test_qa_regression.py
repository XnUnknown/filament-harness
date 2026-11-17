"""Regression tests added during QA review."""
import sys
import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.context import ContextBudget
from filament.files import append_code, replace_line
from filament.llm import MockLLM
from filament.review import review_file
from filament.session import (
    _safe_session_id,
    create_session,
    delete_session,
    load_session,
    save_session,
)
from filament.tools import Toolset


class _CapturingFixClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.reply



class TestQARegressions(unittest.TestCase):
    def test_context_budget_preserves_context_when_marker_does_not_fit(self):
        budget = ContextBudget(max_tokens=8, chars_per_token=4)
        self.assertTrue(budget.add("a" * 4))
        before = budget.dump()
        self.assertFalse(budget.add("x" * 1000))
        self.assertEqual(budget.dump(), before)

    def test_append_code_rejects_drive_relative_path(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                append_code("C:evil.py", "x\n")

    def test_replace_line_rejects_drive_relative_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.py"
            path.write_text("old\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                replace_line("C:evil.py", 1, "new")

    def test_session_id_rejects_colon(self):
        with self.assertRaises(ValueError):
            _safe_session_id("session:evil")


    def test_task_whitespace_target_inherits_parent_target(self):
        from filament.tasks import flatten_tasks, parse_tasks

        tasks = parse_tasks(
            [
                {
                    "id": "parent",
                    "task": "parent work",
                    "target": "parent.py",
                    "subtasks": [{"id": "child", "task": "child work", "target": "   "}],
                }
            ]
        )
        flat = flatten_tasks(tasks)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0].target, "parent.py")

    def test_subtasks_must_be_a_list(self):
        from filament.tasks import parse_tasks

        for bad in ("build the thing", {"task": "nested"}, 5):
            with self.assertRaises(ValueError):
                parse_tasks([{"task": "parent", "subtasks": bad}])

    def test_review_truncation_reports_absolute_line_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "big.py"
            lines = [f"line_{i:04d} = {i}" for i in range(1, 1201)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            client = _CapturingFixClient("FIX 1: replace this")
            fixes = review_file(
                client, path, ContextBudget(max_tokens=2048, chars_per_token=4)
            )
            self.assertEqual(len(fixes), 1)
            line_no, instruction = fixes[0]
            self.assertEqual(instruction, "replace this")
            self.assertGreater(line_no, 1)
            excerpt = client.prompts[0].split("```\n", 1)[1]
            first_excerpt_line = excerpt.splitlines()[0]
            self.assertEqual(lines[line_no - 1], first_excerpt_line)

    def test_toolset_without_shell_runs_unquoted_executable_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig(use_shell=False)
            toolset = Toolset(
                client=MockLLM(),
                budget=ContextBudget(max_tokens=4096),
                config=cfg,
                shell_cwd=d,
            )
            # An unquoted native path exercises backslash handling on Windows;
            # POSIX-mode shlex would otherwise turn C:\dir into C:dir.
            command = f'{sys.executable} -c "print(424242)"'
            result = toolset.run_shell(command, timeout=60)
            self.assertIn("exit_code: 0", result)
            self.assertIn("424242", result)


class TestSessionLifecycleRegressions(unittest.TestCase):
    """Session files must not leak child transcripts or a stale run status."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.directory = self.root / ".filament" / "sessions"

    def tearDown(self):
        self._tmp.cleanup()

    def test_deleting_a_parent_removes_its_subagent_sessions(self):
        parent = create_session(self.root)
        child = create_session(self.root, parent_id=parent.session_id, kind="subagent")
        grandchild = create_session(self.root, parent_id=child.session_id, kind="subagent")
        parent.child_sessions = [child.session_id]
        child.child_sessions = [grandchild.session_id]
        for session in (parent, child, grandchild):
            save_session(session, self.root)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 3)

        self.assertTrue(delete_session(parent.session_id, self.root))

        # Child sessions are hidden from the session list, so a delete that
        # left them behind put them beyond any cleanup the UI could offer.
        self.assertEqual(list(self.directory.glob("*.json")), [])

    def test_a_forged_child_link_does_not_delete_an_unrelated_session(self):
        parent = create_session(self.root)
        other = create_session(self.root)
        parent.child_sessions = [other.session_id]
        save_session(parent, self.root)
        save_session(other, self.root)

        self.assertTrue(delete_session(parent.session_id, self.root))

        self.assertTrue((self.directory / f"{other.session_id}.json").exists())

    def test_a_corrupt_child_cycle_stops(self):
        first = create_session(self.root)
        second = create_session(self.root, parent_id=first.session_id, kind="subagent")
        # A hand-edited pair of files that each claim the other as its child.
        first.child_sessions = [second.session_id]
        second.child_sessions = [first.session_id]
        save_session(first, self.root)
        save_session(second, self.root)

        self.assertTrue(delete_session(first.session_id, self.root))

        self.assertEqual(list(self.directory.glob("*.json")), [])

    def test_a_persisted_running_status_loads_as_idle(self):
        session = create_session(self.root)
        session.status = "running"
        save_session(session, self.root)

        loaded = load_session(session.session_id, self.root)

        # The process that wrote "running" is gone; nothing can join that run.
        self.assertEqual(loaded.status, "idle")


class TestEditingAndContextRegressions(unittest.TestCase):
    """Defects found after the 0.5.8 QA round."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = HarnessConfig()
        self.config.validate()
        from filament.toolkit import ToolRegistry

        self.registry = ToolRegistry(self.root, self.config)

    def tearDown(self):
        self._tmp.cleanup()

    def test_edit_keeps_mixed_line_endings_of_untouched_lines(self):
        path = self.root / "mixed.txt"
        path.write_bytes(b"a\r\nb\nc\r\n")
        self.registry.execute(
            "edit_file", {"path": "mixed.txt", "old_string": "a", "new_string": "A"}
        )
        # Only line 1 changed; the LF-encoded line 2 must not be rewritten as
        # CRLF just because CRLF dominates the file.
        self.assertEqual(path.read_bytes(), b"A\r\nb\nc\r\n")

    def test_edit_inserts_the_file_line_ending(self):
        path = self.root / "win.py"
        path.write_bytes(b"one\r\ntwo\r\n")
        self.registry.execute(
            "edit_file",
            {"path": "win.py", "old_string": "one\r\ntwo", "new_string": "one\nTWO"},
        )
        self.assertEqual(path.read_bytes(), b"one\r\nTWO\r\n")

    def test_edit_matches_the_last_line_without_a_final_newline(self):
        path = self.root / "noeol.py"
        path.write_bytes(b"x = 1")
        out = self.registry.execute(
            "edit_file",
            {"path": "noeol.py", "old_string": "x = 1\n", "new_string": "x = 2\n"},
        )
        self.assertIn("matched the last line", out)
        # The file must keep its missing final newline.
        self.assertEqual(path.read_bytes(), b"x = 2")

    def test_edit_with_a_trailing_newline_does_not_join_a_mid_file_line(self):
        path = self.root / "mid.py"
        path.write_bytes(b"a\nb")
        self.registry.execute(
            "edit_file", {"path": "mid.py", "old_string": "a\n", "new_string": "A\n"}
        )
        # The needle is not at EOF, so its newline must survive the edit.
        self.assertEqual(path.read_bytes(), b"A\nb")

    def test_read_file_flags_a_line_that_cannot_be_paged(self):
        (self.root / "bundle.js").write_text("var a = 1;" + "x" * 100_000, encoding="utf-8")
        out = self.registry.execute("read_file", {"path": "bundle.js"})
        self.assertIn("longer than the 60000-char read limit", out)
        self.assertLess(len(out), 61_000)

    def test_grep_prints_an_overlapping_context_line_once(self):
        (self.root / "dup.txt").write_text(
            "alpha\nneedle1\nneedle2\nomega\n", encoding="utf-8"
        )
        out = self.registry.execute("grep", {"pattern": "needle", "context": 2})
        self.assertEqual(out.count("dup.txt:2:"), 1)
        self.assertEqual(out.count("dup.txt:2-"), 0)
        self.assertEqual(out.count("dup.txt:3:"), 1)
        self.assertEqual(out.count("dup.txt:1-"), 1)

    def test_second_turn_on_one_loop_gets_a_fresh_step_budget(self):
        from filament.agent_loop import AgentLoop
        from filament.session import create_session

        (self.root / "steps.py").write_text("print(1)\n", encoding="utf-8")
        client = MockLLM(
            script=[
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "steps.py"}}]},
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "steps.py"}}]},
                {"content": "third"},
            ]
        )
        cfg = HarnessConfig(max_steps=2)
        cfg.validate()
        loop = AgentLoop(cfg, self.root, create_session(self.root), client=client)
        first = loop.run("first")
        self.assertEqual(first.status, "max_steps")
        second = loop.run("second")
        self.assertEqual(second.status, "done")
        self.assertEqual(second.content, "third")

    def test_env_override_reports_a_named_error_for_an_invalid_llm_table(self):
        from unittest import mock

        from filament.config import load_config

        path = self.root / "filament.toml"
        path.write_text('llm = "oops"\n', encoding="utf-8")
        with mock.patch.dict("os.environ", {"FILAMENT_MODEL": "x"}):
            with self.assertRaises(ValueError) as caught:
                load_config(path)
        self.assertIn("llm must be a table", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
