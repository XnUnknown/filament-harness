import os
import sys
import tempfile
import unittest
from pathlib import Path

from filament.agent_tools import Toolbox
from filament.config import HarnessConfig
from filament.context import ContextBudget
from filament.llm import MockLLM
from filament.shell import prepare_command, split_command
from filament.tools import Toolset


class TestShellCommandValidation(unittest.TestCase):
    def test_toolbox_rejects_null_byte_command(self):
        with tempfile.TemporaryDirectory() as d:
            tools = Toolbox(d, HarnessConfig())
            with self.assertRaises(ValueError):
                tools.run_shell("echo ok" + chr(0))

    def test_toolbox_rejects_control_character_command(self):
        with tempfile.TemporaryDirectory() as d:
            tools = Toolbox(d, HarnessConfig())
            with self.assertRaises(ValueError):
                tools.run_shell("echo ok" + chr(1))

    def test_toolset_rejects_null_byte_command(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig()
            toolset = Toolset(client=MockLLM(), budget=ContextBudget(), config=cfg, shell_cwd=d)
            with self.assertRaises(ValueError):
                toolset.run_shell("echo ok" + chr(0))

    def test_toolset_rejects_control_character_command(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig()
            toolset = Toolset(client=MockLLM(), budget=ContextBudget(), config=cfg, shell_cwd=d)
            with self.assertRaises(ValueError):
                toolset.run_shell("echo ok" + chr(1))

    def test_toolbox_allows_shell_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            tools = Toolbox(d, HarnessConfig(use_shell=True))
            result = tools.run_shell("printf 'a\\nb'")
            self.assertIn("exit_code: 0", result)


    def test_toolset_rejects_non_directory_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            file_cwd = Path(d) / "notdir"
            file_cwd.write_text("x", encoding="utf-8")
            cfg = HarnessConfig()
            toolset = Toolset(client=MockLLM(), budget=ContextBudget(), config=cfg, shell_cwd=str(file_cwd))
            with self.assertRaises(ValueError):
                toolset.run_shell("echo hi")

    def test_toolbox_rejects_non_directory_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            file_cwd = Path(d) / "notdir"
            file_cwd.write_text("x", encoding="utf-8")
            tools = Toolbox(file_cwd, HarnessConfig())
            with self.assertRaises(ValueError):
                tools.run_shell("echo hi")

    def test_toolbox_without_shell_runs_native_executable_path(self):
        # The main orchestrator's Toolbox used bare shlex.split while Toolset
        # used a platform-aware splitter, so shell-less runs broke on Windows
        # paths such as C:\Python\python.exe.
        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig(use_shell=False)
            tools = Toolbox(d, cfg)
            command = f'{sys.executable} -c "print(424242)"'
            result = tools.run_shell(command, timeout=60)
            self.assertIn("exit_code: 0", result)
            self.assertIn("424242", result)

    def test_both_surfaces_share_one_splitter(self):
        # Both tool surfaces must tokenize identically; the Toolbox used to
        # call shlex.split directly and drift from Toolset.
        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig(use_shell=False)
            command = f'{sys.executable} -c "print(777)"'
            toolset = Toolset(client=MockLLM(), budget=ContextBudget(), config=cfg, shell_cwd=d)
            toolbox = Toolbox(d, cfg)
            toolset_result = toolset.run_shell(command, timeout=60)
            toolbox_result = toolbox.run_shell(command, timeout=60)
            self.assertIn("exit_code: 0", toolset_result)
            self.assertIn("exit_code: 0", toolbox_result)
            self.assertIn("777", toolset_result)
            self.assertIn("777", toolbox_result)


class TestCommandSplitting(unittest.TestCase):
    def test_shell_mode_passes_command_through_unchanged(self):
        command = 'echo "a b" | grep a'
        self.assertEqual(prepare_command(command, use_shell=True), command)

    def test_split_preserves_quoted_argument_with_spaces(self):
        self.assertEqual(split_command('python "my script.py" -v'), ["python", "my script.py", "-v"])

    def test_split_rejects_missing_program_name(self):
        with self.assertRaises(ValueError):
            split_command('""')
        with self.assertRaises(ValueError):
            prepare_command('""', use_shell=False)

    def test_split_rejects_non_string_command(self):
        with self.assertRaises(TypeError):
            split_command(None)

    @unittest.skipUnless(os.name == "nt", "backslash paths are native only on Windows")
    def test_split_keeps_windows_backslashes(self):
        self.assertEqual(
            split_command(r"python C:\Users\name\app.py"),
            ["python", r"C:\Users\name\app.py"],
        )


if __name__ == "__main__":
    unittest.main()
