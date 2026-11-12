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


class ToolkitLineEndingTest(unittest.TestCase):
    """Editing a CRLF file must not rewrite every line ending."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = ToolRegistry(self.root, HarnessConfig())

    def tearDown(self):
        self._tmp.cleanup()

    def write_bytes(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_edit_file_preserves_crlf(self):
        path = self.write_bytes("win.py", b"one\r\ntwo\r\nthree\r\n")
        self.registry.execute(
            "edit_file", {"path": "win.py", "old_string": "two\n", "new_string": "TWO\n"}
        )
        self.assertEqual(path.read_bytes(), b"one\r\nTWO\r\nthree\r\n")

    def test_multi_edit_preserves_crlf(self):
        path = self.write_bytes("win.py", b"one\r\ntwo\r\nthree\r\n")
        self.registry.execute(
            "multi_edit",
            {
                "path": "win.py",
                "edits": [
                    {"old_string": "one", "new_string": "ONE"},
                    {"old_string": "three", "new_string": "THREE"},
                ],
            },
        )
        self.assertEqual(path.read_bytes(), b"ONE\r\ntwo\r\nTHREE\r\n")

    def test_write_file_preserves_crlf_on_an_existing_file(self):
        path = self.write_bytes("run.cmd", b"@echo off\r\nrem old\r\n")
        self.registry.execute(
            "write_file", {"path": "run.cmd", "content": "@echo off\nrem new\n"}
        )
        self.assertEqual(path.read_bytes(), b"@echo off\r\nrem new\r\n")

    def test_edit_file_does_not_add_a_trailing_newline(self):
        path = self.write_bytes("noeol.py", b"a\nb")
        self.registry.execute(
            "edit_file", {"path": "noeol.py", "old_string": "a\n", "new_string": "A\n"}
        )
        self.assertEqual(path.read_bytes(), b"A\nb")

    def test_new_file_still_gets_lf_and_a_final_newline(self):
        path = self.root / "new.py"
        self.registry.execute("write_file", {"path": "new.py", "content": "print(1)"})
        self.assertEqual(path.read_bytes(), b"print(1)\n")


class TestFuzzyEditPrecision(unittest.TestCase):
    """The whitespace-tolerant fallback must not rewrite unrelated lines."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = ToolRegistry(self.root, HarnessConfig())

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def test_fuzzy_edit_keeps_trailing_whitespace_on_untouched_lines(self):
        path = self.write("a.py", "alpha   \nbeta\ngamma\t\n")
        out = self.registry.execute(
            "edit_file",
            {"path": "a.py", "old_string": "alpha\nbeta", "new_string": "alpha\nBETA"},
        )
        self.assertIn("matched after normalizing", out)
        self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nBETA\ngamma\t\n")

    def test_fuzzy_edit_on_a_crlf_file_keeps_crlf(self):
        path = self.write("win.py", "one   \r\ntwo\r\nthree\r\n")
        self.registry.execute(
            "edit_file", {"path": "win.py", "old_string": "one\ntwo", "new_string": "one\nTWO"}
        )
        self.assertEqual(path.read_bytes(), b"one\r\nTWO\r\nthree\r\n")

    def test_fuzzy_edit_requires_replace_all_for_repeated_matches(self):
        self.write("a.py", "v\nv\n")
        with self.assertRaises(ToolError):
            self.registry.execute(
                "edit_file", {"path": "a.py", "old_string": "v ", "new_string": "w"}
            )

    def test_fuzzy_edit_replace_all_counts_every_match(self):
        path = self.write("a.py", "v\nv\n")
        out = self.registry.execute(
            "edit_file",
            {"path": "a.py", "old_string": "v ", "new_string": "w", "replace_all": True},
        )
        self.assertIn("2+ replacement", out)
        self.assertEqual(path.read_text(encoding="utf-8"), "w\nw\n")

    def test_whitespace_only_old_string_is_rejected(self):
        path = self.write("a.py", "alpha\nbeta\n")
        with self.assertRaises(ToolError):
            self.registry.execute(
                "edit_file", {"path": "a.py", "old_string": "   ", "new_string": "x"}
            )
        self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nbeta\n")


if __name__ == "__main__":
    unittest.main()
