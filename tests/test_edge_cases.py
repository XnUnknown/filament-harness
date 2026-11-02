import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.context import ContextBudget
from filament.files import extract_code
from filament.llm import MockLLM
from filament.tools import Toolset


class TestEdgeCases(unittest.TestCase):
    def test_extract_code_handles_crlf_fence(self):
        text = "```python\r\nprint(1)\r\n```\r\n"
        self.assertEqual(extract_code(text), "print(1)\n")

    def test_spawn_agent_refuses_when_budget_full(self):
        cfg = HarnessConfig(max_tokens=1024)
        budget = ContextBudget(max_tokens=cfg.effective_max_tokens, chars_per_token=4)
        budget.add("a" * (cfg.effective_max_tokens * 4))
        toolset = Toolset(client=MockLLM(), budget=budget, config=cfg, shell_cwd=".")
        with self.assertRaises(RuntimeError):
            toolset.spawn_agent("Write a Python function named add(a, b).")

    def test_replace_line_roundtrip_no_trailing_newline(self):
        from filament.files import replace_line
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.py"
            path.write_text("old", encoding="utf-8")
            replace_line(path, 1, "new")
            self.assertEqual(path.read_text(encoding="utf-8"), "new")

    def test_flatten_skips_disabled_subtasks(self):
        from filament.tasks import flatten_tasks, parse_tasks

        tasks = parse_tasks(
            [
                {
                    "id": "parent",
                    "task": "parent work",
                    "subtasks": [
                        {"id": "on", "task": "enabled", "enabled": True},
                        {"id": "off", "task": "disabled", "enabled": False},
                    ],
                }
            ]
        )
        self.assertEqual([task.id for task in flatten_tasks(tasks)], ["on"])


if __name__ == "__main__":
    unittest.main()
