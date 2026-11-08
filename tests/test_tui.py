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

    def test_dump_and_load_roundtrip_special_chars(self):
        from filament.config import HarnessConfig

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "filament.toml"
            config = HarnessConfig()
            config.llm.kind = "openai"
            config.llm.model = "a'b\"c"
            config.llm.api_key = "k'ey\"x"
            config.command_allowlist = ["echo 'x'", 'say "hi"']
            _dump_config(path, config)
            loaded = load_config(path)
            self.assertEqual(loaded.llm.model, "a'b\"c")
            self.assertEqual(loaded.llm.api_key, "k'ey\"x")
            self.assertEqual(loaded.command_allowlist, ["echo 'x'", 'say "hi"'])


if __name__ == "__main__":
    unittest.main()
