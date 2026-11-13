"""QA regressions for byte-preserving legacy file writes (Windows line endings)."""
import tempfile
import unittest
from pathlib import Path

from filament.files import append_code, replace_line


class FileLineEndingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_bytes(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_replace_line_keeps_crlf_on_every_line(self):
        # Reading with Path.read_text translated CRLF to LF, so writing one
        # fixed line back converted the whole file to LF.
        path = self.write_bytes("win.py", b"one\r\ntwo\r\nthree\r\n")
        replace_line(path, 2, "TWO")
        self.assertEqual(path.read_bytes(), b"one\r\nTWO\r\nthree\r\n")

    def test_replace_line_keeps_a_missing_final_newline(self):
        path = self.write_bytes("tail.py", b"one\r\ntwo")
        replace_line(path, 2, "TWO")
        self.assertEqual(path.read_bytes(), b"one\r\nTWO")

    def test_append_code_keeps_crlf(self):
        path = self.write_bytes("win.py", b"one\r\ntwo\r\n")
        append_code(path, "three\n", atomic=True)
        self.assertEqual(path.read_bytes(), b"one\r\ntwo\r\nthree\r\n")

    def test_append_code_non_atomic_keeps_crlf(self):
        path = self.write_bytes("win.py", b"one\r\n")
        append_code(path, "two\n")
        self.assertEqual(path.read_bytes(), b"one\r\ntwo\r\n")

    def test_append_code_does_not_glue_onto_a_missing_newline(self):
        for atomic in (False, True):
            with self.subTest(atomic=atomic):
                path = self.write_bytes(f"glue-{atomic}.py", b"one")
                append_code(path, "two\n", atomic=atomic)
                self.assertEqual(path.read_bytes(), b"one\ntwo\n")

    def test_append_code_creates_a_new_file_with_lf(self):
        path = self.root / "new.py"
        append_code(path, "print(1)")
        self.assertEqual(path.read_bytes(), b"print(1)\n")


if __name__ == "__main__":
    unittest.main()
