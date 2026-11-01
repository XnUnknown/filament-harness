import json
import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.orchestrator import Orchestrator


class TestHarness(unittest.TestCase):
    def test_run_batches_and_bounds_context(self):
        with tempfile.TemporaryDirectory() as d:
            target = "out.py"
            target_path = Path(d) / target
            orch = Orchestrator(client_kind="mock", config=HarnessConfig(), shell_cwd=d)
            tasks = [
                "Write a Python function named add(a, b) that returns a + b.",
                "Write a Python function named mul(a, b) that returns a * b.",
            ]
            result = orch.run(tasks, target, review=False)
            self.assertTrue(target_path.exists())
            text = target_path.read_text(encoding="utf-8")
            self.assertIn("def", text)
            self.assertLessEqual(result["context_tokens"], 120000)
            self.assertEqual(result["appended"], 2)


if __name__ == "__main__":
    unittest.main()
