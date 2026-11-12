import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.toolkit import AskUser, FinishTask, ToolError, ToolRegistry


class ToolkitTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = HarnessConfig()
        self.registry = ToolRegistry(self.root, self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")


class TestFileTools(ToolkitTestCase):
    def test_read_file_returns_numbered_lines_and_offsets(self):
        self.write("src/app.py", "one\ntwo\nthree\nfour\n")
        out = self.registry.execute("read_file", {"path": "src/app.py", "offset": 2, "limit": 2})
        self.assertIn("     2  two", out)
        self.assertIn("     3  three", out)
        self.assertNotIn("one", out)
        self.assertIn("offset=4", out)

    def test_read_file_truncation_reports_the_lines_it_actually_showed(self):
        body = "\n".join(f"line_{index:04d} " + "x" * 180 for index in range(1, 401))
        self.write("big.txt", body + "\n")
        out = self.registry.execute("read_file", {"path": "big.txt"})
        self.assertIn("read output truncated", out)
        match = re.search(r"showing 1-(\d+)", out)
        if match is None:
            self.fail("footer did not report the shown line range")
        shown = int(match.group(1))
        self.assertLess(shown, 400)
        self.assertGreaterEqual(shown, 1)
        for number in re.findall(r"^ *(\d+)  ", out, re.MULTILINE):
            self.assertLessEqual(int(number), shown)
        self.assertIn(f"offset={shown + 1}", out)

    def test_write_file_creates_parents_and_tracks_change(self):
        changed = []
        registry = ToolRegistry(self.root, self.config, on_file_change=changed.append)
        out = registry.execute("write_file", {"path": "pkg/mod.py", "content": "x = 1\n"})
        self.assertIn("created pkg/mod.py", out)
        self.assertTrue((self.root / "pkg" / "mod.py").exists())
        self.assertEqual(changed, ["pkg/mod.py"])

    def test_edit_file_replaces_unique_match(self):
        self.write("a.py", "alpha\nbeta\ngamma\n")
        out = self.registry.execute(
            "edit_file", {"path": "a.py", "old_string": "beta", "new_string": "BETA"}
        )
        self.assertIn("-beta", out)
        self.assertIn("+BETA", out)
        self.assertEqual(self.read("a.py"), "alpha\nBETA\ngamma\n")

    def test_edit_file_rejects_ambiguous_match(self):
        self.write("a.py", "x = 1\nx = 2\n")
        with self.assertRaises(ToolError) as caught:
            self.registry.execute(
                "edit_file", {"path": "a.py", "old_string": "x = ", "new_string": "y = "}
            )
        self.assertIn("matches 2 places", str(caught.exception))
        self.assertEqual(self.read("a.py"), "x = 1\nx = 2\n")

    def test_edit_file_replace_all(self):
        self.write("a.py", "x = 1\nx = 2\n")
        self.registry.execute(
            "edit_file",
            {"path": "a.py", "old_string": "x = ", "new_string": "y = ", "replace_all": True},
        )
        self.assertEqual(self.read("a.py"), "y = 1\ny = 2\n")

    def test_edit_file_normalizes_trailing_whitespace(self):
        self.write("a.py", "alpha   \nbeta\n")
        self.registry.execute(
            "edit_file", {"path": "a.py", "old_string": "alpha\nbeta", "new_string": "alpha\nBETA"}
        )
        self.assertEqual(self.read("a.py"), "alpha\nBETA\n")

    def test_edit_file_missing_string_is_an_error(self):
        self.write("a.py", "alpha\n")
        with self.assertRaises(ToolError):
            self.registry.execute(
                "edit_file", {"path": "a.py", "old_string": "nope", "new_string": "x"}
            )

    def test_multi_edit_is_atomic(self):
        self.write("a.py", "one\ntwo\n")
        with self.assertRaises(ToolError):
            self.registry.execute(
                "multi_edit",
                {
                    "path": "a.py",
                    "edits": [
                        {"old_string": "one", "new_string": "ONE"},
                        {"old_string": "missing", "new_string": "X"},
                    ],
                },
            )
        self.assertEqual(self.read("a.py"), "one\ntwo\n")

    def test_multi_edit_applies_all(self):
        self.write("a.py", "one\ntwo\n")
        self.registry.execute(
            "multi_edit",
            {
                "path": "a.py",
                "edits": [
                    {"old_string": "one", "new_string": "ONE"},
                    {"old_string": "two", "new_string": "TWO"},
                ],
            },
        )
        self.assertEqual(self.read("a.py"), "ONE\nTWO\n")

    def test_paths_cannot_escape_the_workspace(self):
        outside = self.root.parent / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        for tool, args in [
            ("read_file", {"path": "../secret.txt"}),
            ("write_file", {"path": "../secret.txt", "content": "x"}),
            ("list_dir", {"path": ".."}),
        ]:
            with self.subTest(tool=tool):
                with self.assertRaises(ToolError):
                    self.registry.execute(tool, args)


class TestSearchTools(ToolkitTestCase):
    def test_grep_finds_lines_and_skips_ignored_dirs(self):
        self.write("src/a.py", "alpha\nneedle = 1\n")
        self.write("src/b.txt", "needle\n")
        self.write("node_modules/x.py", "needle\n")
        out = self.registry.execute("grep", {"pattern": "needle", "path": "."})
        self.assertIn("src/a.py:2:needle = 1", out)
        self.assertIn("src/b.txt:1:needle", out)
        self.assertNotIn("node_modules", out)

    def test_grep_invalid_regex_is_an_error(self):
        with self.assertRaises(ToolError):
            self.registry.execute("grep", {"pattern": "["})

    def test_glob_lists_files_and_rejects_parent_escape(self):
        self.write("a.py", "")
        self.write("pkg/b.py", "")
        out = self.registry.execute("glob", {"pattern": "**/*.py"})
        self.assertIn("a.py", out)
        self.assertIn("pkg/b.py", out)
        with self.assertRaises(ToolError):
            self.registry.execute("glob", {"pattern": "../*.py"})

    def test_glob_works_when_workspace_path_contains_an_ignored_dir_name(self):
        # A checkout under ".../build/project" must not be filtered out just
        # because an ancestor directory shares a name with an ignored folder.
        nested = self.root / "build" / "project"
        (nested / "src").mkdir(parents=True)
        (nested / "app.py").write_text("print(1)\n", encoding="utf-8")
        (nested / "src" / "lib.py").write_text("print(2)\n", encoding="utf-8")
        registry = ToolRegistry(nested, self.config)
        out = registry.execute("glob", {"pattern": "**/*.py"})
        self.assertIn("app.py", out)
        self.assertIn("src/lib.py", out)

    def test_glob_still_skips_ignored_directories_inside_the_workspace(self):
        self.write("src/a.py", "")
        self.write("node_modules/pkg/x.py", "")
        out = self.registry.execute("glob", {"pattern": "**/*.py"})
        self.assertIn("src/a.py", out)
        self.assertNotIn("node_modules", out)

    def test_list_dir_marks_directories(self):
        self.write("pkg/a.py", "")
        out = self.registry.execute("list_dir", {"path": "."})
        self.assertIn("pkg/", out)


class TestShellAndPlanning(ToolkitTestCase):
    def test_run_shell_reports_exit_code(self):
        config = HarnessConfig(use_shell=False)
        registry = ToolRegistry(self.root, config)
        out = registry.execute("run_shell", {"command": f'{sys.executable} -c "print(424242)"'})
        self.assertIn("exit_code: 0", out)
        self.assertIn("424242", out)

    def test_run_shell_honors_allowlist(self):
        config = HarnessConfig(command_allowlist=["echo"])
        registry = ToolRegistry(self.root, config)
        self.assertIn("exit_code: 0", registry.execute("run_shell", {"command": "echo hi"}))
        with self.assertRaises(ToolError):
            registry.execute("run_shell", {"command": "del a.txt"})

    def test_todo_write_validates_status(self):
        out = self.registry.execute(
            "todo_write",
            {"todos": [{"content": "one", "status": "in_progress"}]},
        )
        self.assertIn("[~] one", out)
        self.assertEqual(self.registry.todos[0]["status"], "in_progress")
        with self.assertRaises(ToolError):
            self.registry.execute("todo_write", {"todos": [{"content": "x", "status": "wat"}]})

    def test_ask_user_pauses(self):
        with self.assertRaises(AskUser) as caught:
            self.registry.execute("ask_user", {"question": "Which one?", "options": ["a", "b"]})
        self.assertEqual(caught.exception.question, "Which one?")
        self.assertEqual(caught.exception.options, ["a", "b"])

    def test_finish_raises(self):
        with self.assertRaises(FinishTask) as caught:
            self.registry.execute("finish", {"summary": "done"})
        self.assertEqual(caught.exception.summary, "done")

    def test_plan_mode_hides_mutating_tools(self):
        registry = ToolRegistry(self.root, self.config, read_only=True)
        names = {tool["name"] for tool in registry.specs()}
        self.assertIn("read_file", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("run_shell", names)
        with self.assertRaises(ToolError):
            registry.execute("write_file", {"path": "x.py", "content": "x"})


if __name__ == "__main__":
    unittest.main()
