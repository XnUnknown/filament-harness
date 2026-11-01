import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig, load_config
from filament.files import ensure_target_within_root, extract_code
from filament.review import parse_fixes
from filament.context import ContextBudget
from filament.tasks import flatten_tasks, parse_tasks, task_instruction
from filament.tools import Toolset


class TestReview(unittest.TestCase):
    def test_parse_fixes(self):
        raw = "FIX 3: Missing import\nFix line 5 - rename variable\njunk"
        self.assertEqual(parse_fixes(raw), [(3, "Missing import"), (5, "rename variable")])


class TestContext(unittest.TestCase):
    def test_budget_refuses_overflow(self):
        b = ContextBudget(max_tokens=4, chars_per_token=4)
        self.assertTrue(b.add("a" * 4))
        self.assertFalse(b.add("b" * 200))
        self.assertLessEqual(b.total_tokens(), 4)


class TestShell(unittest.TestCase):
    def test_run_shell_exit_code(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            from filament.config import HarnessConfig
            from filament.llm import MockLLM
            cfg = HarnessConfig()
            toolset = Toolset(client=MockLLM(), budget=ContextBudget(), config=cfg, shell_cwd=d)
            result = toolset.run_shell("printf hello")
            self.assertIn("exit_code: 0", result)
            self.assertIn("hello", result)


class TestConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = HarnessConfig().validate()
        self.assertEqual(cfg.max_tokens, 120000)
        self.assertGreater(cfg.effective_max_tokens, 100000)

    def test_toml_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "filament.toml"
            p.write_text("[llm]\nkind = \"mock\"\nmodel = \"x\"\n", encoding="utf-8")
            cfg = load_config(p)
            self.assertEqual(cfg.llm.kind, "mock")
            self.assertEqual(cfg.llm.model, "x")

    def test_invalid_validation(self):
        with self.assertRaises(ValueError):
            HarnessConfig(max_tokens=10).validate()


class TestTasks(unittest.TestCase):
    def test_subtask_flattening(self):
        tasks = parse_tasks([
            {
                "id": "parent",
                "task": "parent work",
                "subtasks": [
                    "child one",
                    {"id": "c2", "task": "child two"},
                ],
            }
        ])
        flat = flatten_tasks(tasks)
        self.assertEqual([t.id for t in flat], ["parent.1", "c2"])
        self.assertIn("[task parent.1]", task_instruction(flat[0]))

    def test_disabled_and_target(self):
        tasks = parse_tasks([{"task": "skip", "enabled": False}, {"task": "go", "target": "out2.py"}])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].target, "out2.py")


class TestFiles(unittest.TestCase):
    def test_extract_first_fence(self):
        text = "prose\n```python\nprint(1)\n```\n```python\nprint(2)\n```"
        self.assertEqual(extract_code(text), "print(1)\n")

    def test_target_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "allowed").mkdir()
            with self.assertRaises(ValueError):
                ensure_target_within_root(Path(d) / ".." / "outside.py", root, allow_outside=False)
            safe = ensure_target_within_root("allowed/out.py", root, allow_outside=False)
            self.assertTrue(str(safe).endswith("out.py"))


if __name__ == "__main__":
    unittest.main()
