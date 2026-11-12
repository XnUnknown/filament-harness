"""Config type validation: TOML and API updates must fail with named keys."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from filament.config import HarnessConfig, LLMConfig, load_config


class TestTomlTypeValidation(unittest.TestCase):
    def _load(self, body: str) -> HarnessConfig:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "filament.toml"
            path.write_text(body, encoding="utf-8")
            return load_config(path)

    def assert_rejects(self, body: str, *fragments: str) -> str:
        with self.assertRaises(ValueError) as caught:
            self._load(body)
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def test_string_integer_is_rejected_not_type_error(self):
        message = self.assert_rejects('max_tokens = "120000"\n', "max_tokens", "must be an integer")
        self.assertIn("invalid configuration", message)

    def test_quoted_boolean_never_acts_like_true(self):
        # "false" is truthy in Python; it must be rejected, not silently enabled.
        self.assert_rejects('allow_outside_target = "false"\n', "allow_outside_target", "true or false")

    def test_integer_stand_in_for_boolean_is_rejected(self):
        self.assert_rejects("use_shell = 0\n", "use_shell", "true or false")

    def test_bool_is_not_accepted_as_integer(self):
        self.assert_rejects("workers = true\n", "workers", "must be an integer")

    def test_fractional_timeout_is_rejected(self):
        self.assert_rejects("timeout_shell = 1.5\n", "timeout_shell", "must be an integer")

    def test_whole_float_is_normalized_to_int(self):
        cfg = self._load("max_tokens = 120000.0\nworkers = 3.0\n")
        self.assertEqual(cfg.max_tokens, 120000)
        self.assertIsInstance(cfg.max_tokens, int)
        self.assertIsInstance(cfg.workers, int)

    def test_margin_must_be_a_number(self):
        self.assert_rejects('budget_margin = "0.05"\n', "budget_margin", "must be a number")

    def test_allowlist_entries_must_be_strings(self):
        self.assert_rejects('command_allowlist = ["git", 3]\n', "command_allowlist")
        self.assert_rejects('command_allowlist = ["git", "  "]\n', "command_allowlist")

    def test_llm_must_be_a_table(self):
        self.assert_rejects('llm = "ollama"\n', "llm must be a table")
        self.assert_rejects('llm = ["ollama"]\n', "llm must be a table")

    def test_llm_field_types_are_checked(self):
        self.assert_rejects("[llm]\nkind = 5\n", "llm.kind")
        self.assert_rejects('[llm]\nmodel = ""\n', "llm.model")
        self.assert_rejects("[llm]\nbase_url = 7\n", "llm.base_url")
        self.assert_rejects('[llm]\nuse_ollama_env = "1"\n', "llm.use_ollama_env")

    def test_top_level_bad_keys_are_reported_together(self):
        message = self.assert_rejects(
            'max_tokens = "1"\nworkers = "4"\natomic_writes = "yes"\n',
            "max_tokens",
            "workers",
            "atomic_writes",
        )
        self.assertGreaterEqual(message.count("must be"), 3)

    def test_valid_config_still_loads(self):
        cfg = self._load(
            "max_tokens = 8192\n"
            "budget_margin = 0.1\n"
            "workers = 2\n"
            "atomic_writes = false\n"
            "command_allowlist = [\"git\", \"python\"]\n"
            "[llm]\nkind = \"mock\"\nmodel = \"demo\"\n"
        )
        self.assertEqual(cfg.workers, 2)
        self.assertFalse(cfg.atomic_writes)
        self.assertIsInstance(cfg.budget_margin, float)
        self.assertEqual(cfg.command_allowlist, ["git", "python"])


class TestDirectValidation(unittest.TestCase):
    def test_validate_normalizes_fields_in_place(self):
        cfg = HarnessConfig(workers=2.0, budget_margin=0)  # type: ignore[arg-type]
        cfg.validate()
        self.assertIsInstance(cfg.workers, int)
        self.assertIsInstance(cfg.budget_margin, float)

    def test_validate_rejects_wrong_llm_kind_type(self):
        cfg = HarnessConfig()
        cfg.llm.kind = 7  # type: ignore[assignment]
        with self.assertRaises(ValueError) as caught:
            cfg.validate()
        self.assertIn("llm.kind", str(caught.exception))

    def test_llm_validate_alone_reports_its_own_keys(self):
        llm = LLMConfig(kind="mock")
        llm.model = None  # type: ignore[assignment]
        with self.assertRaises(ValueError) as caught:
            llm.validate()
        self.assertIn("llm.model", str(caught.exception))


class TestEnvironmentPrecedence(unittest.TestCase):
    def _load_with_env(self, body, env):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "filament.toml"
            if body is not None:
                path.write_text(body, encoding="utf-8")
            with patch.dict(os.environ, env, clear=True):
                return load_config(path)

    def test_explicit_client_beats_the_legacy_ollama_flag(self):
        cfg = self._load_with_env(
            None, {"FILAMENT_CLIENT": "mock", "FILAMENT_USE_OLLAMA": "1"}
        )
        self.assertEqual(cfg.llm.kind, "mock")

    def test_legacy_flag_still_selects_ollama_for_auto(self):
        cfg = self._load_with_env(None, {"FILAMENT_USE_OLLAMA": "1"})
        self.assertEqual(cfg.llm.kind, "ollama")

    def test_toml_provider_beats_the_legacy_flag(self):
        cfg = self._load_with_env(
            '[llm]\nkind = "openai"\n',
            {"FILAMENT_USE_OLLAMA": "1", "FILAMENT_API_KEY": "k"},
        )
        self.assertEqual(cfg.llm.kind, "openai")

    def test_invalid_llm_shape_is_still_a_validation_error(self):
        with self.assertRaises(ValueError) as caught:
            self._load_with_env(
                'llm = "ollama"\n',
                {"FILAMENT_USE_OLLAMA": "1", "OLLAMA_API_KEY": "k"},
            )
        self.assertIn("llm must be a table", str(caught.exception))


class TestBackendConfigParity(unittest.TestCase):
    def test_backend_uses_the_shared_validator(self):
        from filament.backend_server import _apply_config

        cfg = HarnessConfig()
        updated = _apply_config(cfg, {"max_tokens": 2048.0})
        self.assertEqual(updated.max_tokens, 2048)
        self.assertIsInstance(updated.max_tokens, int)
        with self.assertRaises(ValueError):
            _apply_config(cfg, {"timeout_shell": 1.5})
        self.assertEqual(cfg.max_tokens, 120_000)


if __name__ == "__main__":
    unittest.main()
