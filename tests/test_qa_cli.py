"""QA regression tests for CLI parsing and the TUI provider menu."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from filament.cli import _run_batch, build_parser
from filament.config import HarnessConfig
from filament.tui import _prompt_choice


class TestGlobalOptionsBeforeSubcommands(unittest.TestCase):
    def test_shell_cwd_survives_before_run(self):
        args = build_parser().parse_args(["--shell-cwd", "W", "run", "tasks.json"])
        self.assertEqual(getattr(args, "shell_cwd", "."), "W")

    def test_config_survives_before_run(self):
        args = build_parser().parse_args(["--config", "cfg.toml", "run", "tasks.json"])
        self.assertEqual(getattr(args, "config", None), "cfg.toml")

    def test_model_budget_and_client_survive_before_run(self):
        args = build_parser().parse_args(
            [
                "--client",
                "mock",
                "--model",
                "m",
                "--max-tokens",
                "500",
                "--no-stream",
                "run",
                "tasks.json",
            ]
        )
        self.assertEqual(args.client, "mock")
        self.assertEqual(args.model, "m")
        self.assertEqual(args.max_tokens, 500)
        self.assertTrue(args.no_stream)

    def test_run_subcommand_options_still_parse_after_it(self):
        args = build_parser().parse_args(
            ["run", "tasks.json", "--shell-cwd", "W", "--client", "mock", "--max-steps", "3"]
        )
        self.assertEqual(args.shell_cwd, "W")
        self.assertEqual(args.client, "mock")
        self.assertEqual(args.max_steps, 3)

    def test_run_defaults_are_absent_or_safe(self):
        args = build_parser().parse_args(["run", "tasks.json"])
        self.assertEqual(getattr(args, "shell_cwd", "."), ".")
        self.assertIsNone(getattr(args, "config", None))
        self.assertIsNone(getattr(args, "client", None))
        self.assertIsNone(getattr(args, "max_tokens", None))
        self.assertFalse(getattr(args, "no_stream", False))

    def test_tui_config_survives_before_subcommand(self):
        args = build_parser().parse_args(["--config", "X", "tui"])
        self.assertEqual(getattr(args, "config", "filament.toml"), "X")

    def test_tui_defaults_to_filament_toml(self):
        args = build_parser().parse_args(["tui"])
        self.assertEqual(getattr(args, "config", "filament.toml"), "filament.toml")


class TestRunBatchForwardsWorkspace(unittest.TestCase):
    def test_global_shell_cwd_reaches_the_orchestrator(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = Path(directory) / "tasks.json"
            tasks.write_text('[{"task": "write add(a, b)"}]', encoding="utf-8")
            args = build_parser().parse_args(
                ["--shell-cwd", directory, "run", str(tasks), "--client", "mock", "--quiet"]
            )
            with patch("filament.cli.load_config", return_value=HarnessConfig()):
                with patch("filament.orchestrator.Orchestrator") as fake:
                    fake.return_value.run.return_value = {
                        "errors": [],
                        "appended": 0,
                        "syntax_ok": True,
                    }
                    with redirect_stdout(io.StringIO()):
                        code = _run_batch(args)
        self.assertEqual(code, 0)
        self.assertEqual(fake.call_args.kwargs["shell_cwd"], directory)


class TestPromptChoice(unittest.TestCase):
    def test_stale_default_outside_the_menu_does_not_loop(self):
        calls = []

        def prompt(_text, default=""):
            calls.append(default)
            if len(calls) > 1:
                raise AssertionError("_prompt_choice looped on an invalid default")
            return default

        with patch("filament.tui._prompt", side_effect=prompt):
            with redirect_stdout(io.StringIO()):
                choice = _prompt_choice("Provider", ["auto", "mock"], "removed-kind")
        self.assertEqual(choice, "auto")
        self.assertEqual(calls, ["auto"])


if __name__ == "__main__":
    unittest.main()
