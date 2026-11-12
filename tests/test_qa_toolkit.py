"""QA regression tests for tool argument validation."""
import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.toolkit import ToolError, ToolRegistry


class ToolkitBooleanArgumentTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = ToolRegistry(self.root, HarnessConfig())

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_edit_file_string_false_does_not_replace_all(self):
        path = self.write("a.py", "x = 1\nx = 2\n")
        with self.assertRaises(ToolError) as caught:
            self.registry.execute(
                "edit_file",
                {
                    "path": "a.py",
                    "old_string": "x = ",
                    "new_string": "y = ",
                    "replace_all": "false",
                },
            )
        self.assertIn("replace_all", str(caught.exception))
        self.assertEqual(path.read_text(encoding="utf-8"), "x = 1\nx = 2\n")

    def test_edit_file_real_true_still_replaces_all(self):
        path = self.write("a.py", "x = 1\nx = 2\n")
        self.registry.execute(
            "edit_file",
            {
                "path": "a.py",
                "old_string": "x = ",
                "new_string": "y = ",
                "replace_all": True,
            },
        )
        self.assertEqual(path.read_text(encoding="utf-8"), "y = 1\ny = 2\n")

    def test_multi_edit_string_false_is_rejected_atomically(self):
        path = self.write("a.py", "one\ntwo\n")
        with self.assertRaises(ToolError):
            self.registry.execute(
                "multi_edit",
                {
                    "path": "a.py",
                    "edits": [
                        {"old_string": "one", "new_string": "ONE"},
                        {"old_string": "two", "new_string": "TWO", "replace_all": "false"},
                    ],
                },
            )
        self.assertEqual(path.read_text(encoding="utf-8"), "one\ntwo\n")

    def test_grep_string_boolean_is_rejected(self):
        self.write("a.py", "Needle\nneedle\n")
        with self.assertRaises(ToolError) as caught:
            self.registry.execute("grep", {"pattern": "needle", "ignore_case": "false"})
        self.assertIn("ignore_case", str(caught.exception))

    def test_grep_real_boolean_is_honored(self):
        self.write("a.py", "Needle\nneedle\n")
        insensitive = self.registry.execute("grep", {"pattern": "needle", "ignore_case": True})
        sensitive = self.registry.execute("grep", {"pattern": "needle", "ignore_case": False})
        self.assertEqual(insensitive.count("eedle"), 2)
        self.assertEqual(sensitive.count("eedle"), 1)

    def test_spawn_agents_rejects_whitespace_only_string_task(self):
        spawned = []

        def spawn(task, system, call_id):
            spawned.append(task)
            return {"session_id": "s", "status": "done", "steps": 0, "content": "ok"}

        registry = ToolRegistry(self.root, HarnessConfig(), spawn_fn=spawn)
        with self.assertRaises(ToolError) as caught:
            registry.execute("spawn_agents", {"tasks": ["   "]})
        self.assertIn("task 1 is empty", str(caught.exception))
        self.assertEqual(spawned, [])


if __name__ == "__main__":
    unittest.main()
