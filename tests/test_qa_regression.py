"""Regression tests added during QA review."""
import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.context import ContextBudget
from filament.files import append_code, replace_line
from filament.session import _safe_session_id


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

if __name__ == "__main__":
    unittest.main()
