import tempfile
import unittest
from pathlib import Path

from filament.agent_tools import Toolbox
from filament.config import (
    DEFAULT_READ_MAX_CHARS,
    MAX_SHELL_TIMEOUT_SECONDS,
    HarnessConfig,
    coerce_char_limit,
    coerce_max_fixes,
    coerce_timeout,
)
from filament.context import ContextBudget
from filament.llm import MockLLM
from filament.tools import Toolset


class TestCoerceTimeout(unittest.TestCase):
    def test_none_uses_default(self):
        self.assertEqual(coerce_timeout(None, 60), 60.0)

    def test_accepts_positive_number(self):
        self.assertEqual(coerce_timeout(5, 60), 5.0)
        self.assertEqual(coerce_timeout(2.5, 60), 2.5)

    def test_rejects_numeric_string(self):
        with self.assertRaises(ValueError):
            coerce_timeout("10", 60)

    def test_rejects_bool(self):
        with self.assertRaises(ValueError):
            coerce_timeout(True, 60)

    def test_rejects_non_positive(self):
        for value in (0, -1, -0.5):
            with self.assertRaises(ValueError):
                coerce_timeout(value, 60)

    def test_rejects_beyond_upper_bound(self):
        with self.assertRaises(ValueError):
            coerce_timeout(MAX_SHELL_TIMEOUT_SECONDS + 1, 60)


class TestCoerceCharLimit(unittest.TestCase):
    def test_none_uses_default(self):
        self.assertEqual(coerce_char_limit(None), DEFAULT_READ_MAX_CHARS)

    def test_accepts_int_and_numeric_string(self):
        self.assertEqual(coerce_char_limit(40), 40)
        self.assertEqual(coerce_char_limit("40"), 40)

    def test_rejects_non_numeric_string(self):
        with self.assertRaises(ValueError):
            coerce_char_limit("abc")

    def test_rejects_float_and_bool(self):
        for value in (1.5, True):
            with self.assertRaises(ValueError):
                coerce_char_limit(value)

    def test_rejects_non_positive(self):
        for value in (0, -5):
            with self.assertRaises(ValueError):
                coerce_char_limit(value)


class TestCoerceMaxFixes(unittest.TestCase):
    def test_none_uses_default(self):
        self.assertIsNone(coerce_max_fixes(None))
        self.assertEqual(coerce_max_fixes(None, 3), 3)

    def test_accepts_zero_and_positive_int(self):
        self.assertEqual(coerce_max_fixes(0), 0)
        self.assertEqual(coerce_max_fixes(4), 4)

    def test_accepts_whole_float(self):
        self.assertEqual(coerce_max_fixes(2.0), 2)

    def test_rejects_bool_string_fraction_and_negative(self):
        for value in (True, "2", 2.5, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    coerce_max_fixes(value)


class TestShellTimeoutArguments(unittest.TestCase):
    def test_toolbox_rejects_non_numeric_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            tools = Toolbox(d, HarnessConfig())
            with self.assertRaises(ValueError):
                tools.run_shell("echo hi", timeout="10")  # type: ignore[arg-type]

    def test_toolset_rejects_non_numeric_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            toolset = Toolset(client=MockLLM(), budget=ContextBudget(), config=HarnessConfig(), shell_cwd=d)
            with self.assertRaises(ValueError):
                toolset.run_shell("echo hi", timeout="10")  # type: ignore[arg-type]

    def test_toolbox_accepts_valid_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            tools = Toolbox(d, HarnessConfig(use_shell=True))
            self.assertIn("exit_code: 0", tools.run_shell("echo hi", timeout=10))

    def test_call_raises_value_error_for_bad_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            tools = Toolbox(d, HarnessConfig())
            with self.assertRaises(ValueError):
                tools.call("run_shell", {"command": "echo hi", "timeout": "10"})


class TestReadFileCharLimit(unittest.TestCase):
    def _write(self, directory: str) -> str:
        path = Path(directory) / "sample.txt"
        path.write_text("abcdefghij" * 10, encoding="utf-8")
        return "sample.txt"

    def test_numeric_string_limit_truncates(self):
        with tempfile.TemporaryDirectory() as d:
            relative = self._write(d)
            content = Toolbox(d, HarnessConfig()).read_file(relative, max_chars="20")  # type: ignore[arg-type]
            self.assertTrue(content.startswith("abcdefghij" * 2))
            self.assertIn("truncated 80 chars", content)

    def test_call_forwards_string_limit(self):
        with tempfile.TemporaryDirectory() as d:
            relative = self._write(d)
            content = Toolbox(d, HarnessConfig()).call("read_file", {"path": relative, "max_chars": "20"})
            self.assertIn("truncated 80 chars", content)

    def test_zero_and_negative_limits_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            relative = self._write(d)
            tools = Toolbox(d, HarnessConfig())
            for value in (0, -5):
                with self.assertRaises(ValueError):
                    tools.read_file(relative, max_chars=value)

    def test_non_numeric_limit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            relative = self._write(d)
            with self.assertRaises(ValueError):
                Toolbox(d, HarnessConfig()).read_file(relative, max_chars="abc")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
