"""QA regression tests for tool argument validation."""
import os
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

    def test_fuzzy_edit_with_a_trailing_newline_keeps_the_next_line_indented(self):
        # The needle's newline was matched, then `[ \t]*` swallowed the body's
        # indentation, so a header edit de-indented the function it opened.
        path = self.write("win.py", "def f():\r\n    pass\r\n")
        self.registry.execute(
            "edit_file",
            {"path": "win.py", "old_string": "def f():\n", "new_string": "def g():\n"},
        )
        self.assertEqual(path.read_bytes(), b"def g():\r\n    pass\r\n")

    def test_fuzzy_edit_with_a_trailing_newline_keeps_the_body_after_trailing_space(self):
        path = self.write("a.py", "def f(): \n    pass\n")
        out = self.registry.execute(
            "edit_file",
            {"path": "a.py", "old_string": "def f():\n", "new_string": "def g():\n"},
        )
        self.assertIn("matched after normalizing", out)
        self.assertEqual(path.read_bytes(), b"def g():\n    pass\n")

    def test_fuzzy_edit_with_a_trailing_newline_still_requires_the_line_end(self):
        # A line prefix must not match when the needle asked for a whole line.
        path = self.write("a.py", "foobar\n")
        with self.assertRaises(ToolError):
            self.registry.execute(
                "edit_file", {"path": "a.py", "old_string": "foo\n", "new_string": "X\n"}
            )
        self.assertEqual(path.read_text(encoding="utf-8"), "foobar\n")


class ToolkitCheckPythonEncodingTest(unittest.TestCase):
    def test_non_utf8_file_is_reported_as_a_tool_error(self):
        with tempfile.TemporaryDirectory() as d:
            registry = ToolRegistry(Path(d), HarnessConfig())
            (Path(d) / "latin.py").write_bytes(b"# caf\xe9\nx = 1\n")
            with self.assertRaises(ToolError) as caught:
                registry.execute("check_python", {"path": "latin.py"})
            self.assertIn("not valid UTF-8", str(caught.exception))


class ToolkitGrepConfinementTest(unittest.TestCase):
    def test_confined_path_rejects_a_resolved_path_outside_the_workspace(self):
        with tempfile.TemporaryDirectory() as outside, tempfile.TemporaryDirectory() as inside:
            registry = ToolRegistry(Path(inside), HarnessConfig())
            outside_file = Path(outside) / "secret.txt"
            outside_file.write_text("leaked-token\n", encoding="utf-8")
            inside_file = Path(inside) / "ok.txt"
            inside_file.write_text("fine\n", encoding="utf-8")
            self.assertIsNone(registry._confined_path(outside_file))
            self.assertEqual(registry._confined_path(inside_file), inside_file.resolve())

            permissive = ToolRegistry(Path(inside), HarnessConfig(allow_outside_target=True))
            self.assertEqual(permissive._confined_path(outside_file), outside_file.resolve())

    def test_grep_does_not_read_through_a_symlink_outside_the_workspace(self):
        with tempfile.TemporaryDirectory() as outside, tempfile.TemporaryDirectory() as inside:
            secret = Path(outside) / "secret.txt"
            secret.write_text("leaked-token\n", encoding="utf-8")
            try:
                os.symlink(secret, Path(inside) / "link.txt")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are not available on this host")
            registry = ToolRegistry(Path(inside), HarnessConfig())
            self.assertIn("no matches", registry.execute("grep", {"pattern": "leaked-token"}))
            with self.assertRaises(ToolError):
                registry.execute("read_file", {"path": "link.txt"})


class ShellStdinInheritanceTest(unittest.TestCase):
    """A shell tool must not read the harness's own stdin.

    In ``filament serve`` stdin is the newline-delimited JSON protocol.  A
    command that reads stdin (``python script.py``, ``cat``, ``git commit``
    without ``-m``) inherited that pipe and consumed request lines, so the
    matching replies never arrived and the UI hung.
    """

    MARKER = "REQUEST LINE THAT MUST SURVIVE"

    def _run_helper(self, helper_code):
        import subprocess
        import sys

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper.py"
            helper.write_text(
                f"import sys\nsys.path.insert(0, {str(project_root)!r})\n" + helper_code,
                encoding="utf-8",
            )
            return subprocess.run(
                [sys.executable, str(helper), str(root)],
                input=(self.MARKER + "\n").encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )

    @staticmethod
    def _reader_command():
        import sys

        program = "import sys; data = sys.stdin.read(); print('CHILD_SAW', len(data))"
        return f'{sys.executable} -c "{program}"'

    def _assert_marker_survived(self, setup, invoke):
        command = self._reader_command()
        body = (
            "from pathlib import Path\n"
            "from filament.config import HarnessConfig\n"
            + setup.replace("{command}", repr(command))
            + invoke.replace("{command}", repr(command))
            + "\nprint('PARENT_REMAINING=' + repr(sys.stdin.read()))\n"
        )
        proc = self._run_helper(body)
        output = proc.stdout.decode("utf-8", "replace")
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        self.assertIn("CHILD_SAW 0", output)
        self.assertIn("PARENT_REMAINING=" + repr(self.MARKER + "\n"), output)

    def test_the_agent_shell_tool_does_not_consume_harness_stdin(self):
        self._assert_marker_survived(
            "from filament.toolkit import ToolRegistry\n"
            "config = HarnessConfig()\nconfig.validate()\n"
            "registry = ToolRegistry(Path(sys.argv[1]), config)\n",
            "print(registry.execute('run_shell', {'command': {command}, 'timeout': 30}))\n",
        )

    def test_the_backend_toolbox_shell_does_not_consume_harness_stdin(self):
        self._assert_marker_survived(
            "from filament.agent_tools import Toolbox\n"
            "config = HarnessConfig()\nconfig.validate()\n"
            "registry = Toolbox(Path(sys.argv[1]), config)\n",
            "print(registry.run_shell({command}, timeout=30))\n",
        )


if __name__ == "__main__":
    unittest.main()
