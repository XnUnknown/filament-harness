import tempfile
import unittest
from pathlib import Path

from filament.agent_tools import Toolbox
from filament.config import HarnessConfig
from filament.context import ContextBudget
from filament.llm import MockLLM
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


if __name__ == "__main__":
    unittest.main()
