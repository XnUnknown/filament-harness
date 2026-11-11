import unittest

from filament.cli import build_parser


class TestCLIParser(unittest.TestCase):
    def test_default_chat_accepts_global_options(self):
        args = build_parser().parse_args(["--client", "mock", "--shell-cwd", "workspace"])
        self.assertIsNone(args.cmd)
        self.assertEqual(args.client, "mock")
        self.assertEqual(args.shell_cwd, "workspace")

    def test_chat_subcommand_defaults(self):
        args = build_parser().parse_args(["chat"])
        self.assertEqual(args.cmd, "chat")
        self.assertEqual(getattr(args, "shell_cwd", "."), ".")
        self.assertEqual(getattr(args, "mode", "build"), "build")
        self.assertEqual(getattr(args, "config", "filament.toml"), "filament.toml")
        self.assertFalse(getattr(args, "no_stream", False))

    def test_global_config_survives_before_subcommand(self):
        args = build_parser().parse_args(["--config", "X", "chat"])
        self.assertEqual(args.config, "X")

    def test_chat_option_overrides_global(self):
        args = build_parser().parse_args(["--config", "X", "chat", "--config", "Y"])
        self.assertEqual(args.config, "Y")

    def test_chat_subcommand_overrides(self):
        args = build_parser().parse_args(["chat", "--mode", "plan", "--max-steps", "5"])
        self.assertEqual(args.mode, "plan")
        self.assertEqual(args.max_steps, 5)

    def test_run_subcommand_is_unaffected(self):
        args = build_parser().parse_args(["run", "tasks.json", "--client", "mock", "--quiet"])
        self.assertEqual(args.cmd, "run")
        self.assertEqual(args.tasks_file, "tasks.json")
        self.assertTrue(args.quiet)

    def test_version_flag_is_registered(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--version"])


if __name__ == "__main__":
    unittest.main()
