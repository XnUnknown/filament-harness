"""write_file must report the file it actually wrote.

Two reporting defects made the tool result disagree with the disk:

* ``content.count("\\n") + 1`` counted a trailing newline as a second line, so
  every normal file ("print(1)\\n") was reported as two lines.
* ``_write_text`` appended a newline to *every* missing final newline, including
  an empty file, so ``write_file(path, "")`` created a one-byte file that could
  not be used as an empty placeholder (for example ``__init__.py``).

These tests pin the file bytes and the count in the tool result together.
"""
import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.toolkit import ToolRegistry


class TestWriteFileReporting(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = ToolRegistry(self.root, HarnessConfig())

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, content):
        result = self.registry.execute(
            "write_file", {"path": name, "content": content}
        )
        return (self.root / name).read_bytes(), result

    def test_empty_content_creates_an_empty_file(self):
        data, result = self.write("empty.py", "")
        self.assertEqual(data, b"")
        self.assertIn("0 chars", result)
        self.assertIn("0 lines", result)

    def test_overwriting_with_empty_content_clears_the_file(self):
        (self.root / "notes.txt").write_bytes(b"old body\n")
        data, result = self.write("notes.txt", "")
        self.assertEqual(data, b"")
        self.assertIn("overwrote", result)

    def test_trailing_newline_is_not_counted_as_an_extra_line(self):
        data, result = self.write("one.py", "print(1)\n")
        self.assertEqual(data, b"print(1)\n")
        self.assertIn("1 line)", result)
        self.assertNotIn("2 lines", result)

    def test_line_count_for_a_file_without_a_final_newline(self):
        data, result = self.write("two.py", "a\nb")
        self.assertEqual(data, b"a\nb\n")
        self.assertIn("2 lines", result)

    def test_line_count_keeps_blank_trailing_lines(self):
        data, result = self.write("blank.py", "a\n\n")
        self.assertEqual(data, b"a\n\n")
        self.assertIn("2 lines", result)


if __name__ == "__main__":
    unittest.main()
