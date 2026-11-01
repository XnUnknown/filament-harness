import tempfile
import unittest
from pathlib import Path

from filament.config import load_config
from filament.tui import _dump_config


class TestTUIConfig(unittest.TestCase):
    def test_dump_and_load_roundtrip(self):
        from filament.config import HarnessConfig

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "filament.toml"
            config = HarnessConfig()
            config.llm.kind = "mock"
            config.llm.model = "demo-model"
            config.command_allowlist = ["echo", "python"]
            _dump_config(path, config)
            loaded = load_config(path)
            self.assertEqual(loaded.llm.kind, "mock")
            self.assertEqual(loaded.llm.model, "demo-model")
            self.assertEqual(loaded.command_allowlist, ["echo", "python"])
            self.assertEqual(loaded.max_tokens, config.max_tokens)


if __name__ == "__main__":
    unittest.main()
