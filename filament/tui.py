"""Interactive terminal UI for Filament.

The chat screen is the primary surface and uses the same agent loop and console
renderer as the CLI.  The menus remain for editing provider, context, and
execution settings without touching TOML by hand.  Standard library only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import HarnessConfig, load_config
from .console import AgentConsole, ConsoleRenderer


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _line(char: str = "-", width: int = 72) -> str:
    return char * width


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{text}{suffix}: ").strip()
    except EOFError:
        return default
    return value if value else default


def _prompt_int(text: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = _prompt(text, str(default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            print("Please enter an integer.")
            continue
        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        return value


def _prompt_bool(text: str, default: bool) -> bool:
    raw = _prompt(f"{text} (y/n)", "y" if default else "n")
    return raw.lower() in {"y", "yes", "true", "1"}


def _prompt_choice(text: str, choices: list[str], default: str) -> str:
    if default not in choices:
        # A config value such as kind="auto" is not in this menu; falling back
        # to the first choice keeps Enter from looping on "Invalid choice".
        default = choices[0]
    print(text)
    for index, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {index}. {choice}{marker}")
    while True:
        raw = _prompt("Choice", default)
        if raw in choices:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print("Invalid choice.")


def _dump_config(path: Path, config: HarnessConfig) -> None:
    llm = config.llm
    lines = [
        f"max_tokens = {config.max_tokens}",
        f"chars_per_token = {config.chars_per_token}",
        f"budget_margin = {config.budget_margin}",
        f"workers = {config.workers}",
        f"max_tasks = {config.max_tasks}",
        f"timeout_shell = {config.timeout_shell}",
        f"max_shell_output = {config.max_shell_output}",
        f"allow_outside_target = {str(config.allow_outside_target).lower()}",
        f"atomic_writes = {str(config.atomic_writes).lower()}",
        f"use_shell = {str(config.use_shell).lower()}",
        f"offload_shell_output = {str(config.offload_shell_output).lower()}",
        f"max_steps = {config.max_steps}",
        f"subagent_max_steps = {config.subagent_max_steps}",
        f"subagent_max_tokens = {config.subagent_max_tokens}",
        f"tool_output_limit = {config.tool_output_limit}",
        f"keep_recent_messages = {config.keep_recent_messages}",
        f"compact_threshold = {config.compact_threshold}",
        f"compact_with_llm = {str(config.compact_with_llm).lower()}",
        f"max_depth = {config.max_depth}",
        f"command_allowlist = {json.dumps(config.command_allowlist)}",
        "",
        "[llm]",
        f"kind = {json.dumps(llm.kind)}",
        f"model = {json.dumps(llm.model)}",
        f"base_url = {json.dumps(llm.base_url or '')}",
        f"api_key = {json.dumps(llm.api_key or '')}",
        f"timeout = {llm.timeout}",
        f"stream = {str(llm.stream).lower()}",
        f"max_output_tokens = {llm.max_output_tokens}",
    ]
    if llm.reasoning_effort:
        lines.insert(-1, f"reasoning_effort = {json.dumps(llm.reasoning_effort)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TUIApp:
    def __init__(self, config_path: str | Path = "filament.toml"):
        self.config_path = Path(config_path)
        try:
            self.config = load_config(self.config_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load config ({exc}); using defaults.")
            self.config = HarnessConfig()
        self.tasks: list[str] = []
        self.shell_cwd = "."
        self.target = "output.py"
        self.allow_outside_target = self.config.allow_outside_target
        self.review = True
        self.fix = True
        self.max_fixes: int | None = None

    # -- menus --------------------------------------------------------------
    def main_menu(self) -> None:
        while True:
            _clear()
            print("Filament Harness")
            print(_line())
            print(f"Config: {self.config_path} | Provider: {self.config.llm.kind} | Model: {self.config.llm.model}")
            print(f"Workspace: {self.shell_cwd} | Tasks: {len(self.tasks)} | Target: {self.target}")
            print(_line())
            print("1. Chat with the agent")
            print("2. Run a batch of tasks (legacy)")
            print("3. Manage batch tasks")
            print("4. Provider settings")
            print("5. Context and agent settings")
            print("6. Execution and security settings")
            print("7. Paths and defaults")
            print("8. Load config")
            print("9. Save config")
            print("0. Show current config")
            print("q. Quit")
            choice = _prompt("Select", "1")
            if choice == "1":
                self.chat_mode()
            elif choice == "2":
                self.run_tasks()
            elif choice == "3":
                self.manage_tasks()
            elif choice == "4":
                self.provider_settings()
            elif choice == "5":
                self.context_settings()
            elif choice == "6":
                self.execution_settings()
            elif choice == "7":
                self.path_settings()
            elif choice == "8":
                self.load_config()
            elif choice == "9":
                self.save_config()
            elif choice == "0":
                self.show_config()
                _prompt("Press Enter to continue", "")
            elif choice.lower() in {"q", "quit", "exit"}:
                return
            else:
                print("Unknown choice.")
                _prompt("Press Enter to continue", "")

    def chat_mode(self) -> None:
        from .chat import ChatOrchestrator
        from .cli import _handle_command

        chat = ChatOrchestrator(self.config, self.shell_cwd, mode="build")
        console = AgentConsole(chat)
        _clear()
        renderer = ConsoleRenderer()
        renderer._line(renderer._paint("filament", "33") + f"  {self.config.llm.kind}/{self.config.llm.model}")
        renderer._line(renderer._paint(f"workspace {chat.shell_cwd}  session {chat.session.session_id}", "2"))
        renderer._line(renderer._paint("commands: /new /sessions /load /mode /compress /todos /context /quit", "2"))
        while True:
            try:
                message = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message.startswith("/") or message.lower() in {"exit", "quit"}:
                if not _handle_command(chat, message):
                    break
                continue
            try:
                result = console.run_turn(message)
            except KeyboardInterrupt:
                pending = chat.session.pending_question
                chat.cancel()
                if pending:
                    print("\n[question dismissed; the next message starts a fresh turn]")
                else:
                    print("\n[cancelling after the current model call]")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"error: {exc}")
                continue
            console.print_result(result)
        chat.save()

    def manage_tasks(self) -> None:
        while True:
            _clear()
            print("Batch Tasks")
            print(_line())
            if not self.tasks:
                print("(no tasks)")
            for index, task in enumerate(self.tasks, 1):
                print(f"{index}. {task}")
            print(_line())
            print("a. Add task")
            print("r. Remove task")
            print("c. Clear tasks")
            print("b. Back")
            choice = _prompt("Select", "b")
            if choice == "a":
                task = _prompt("Task instruction")
                if task:
                    self.tasks.append(task)
            elif choice == "r":
                if self.tasks:
                    number = _prompt_int("Task number to remove", len(self.tasks), 1)
                    if 1 <= number <= len(self.tasks):
                        self.tasks.pop(number - 1)
            elif choice == "c":
                self.tasks.clear()
            elif choice == "b":
                return
            else:
                print("Unknown choice.")
                _prompt("Press Enter to continue", "")

    def provider_settings(self) -> None:
        _clear()
        print("Provider Settings")
        print(_line())
        current = self.config.llm.kind
        kind = _prompt_choice("Provider", ["auto", "mock", "ollama", "openai", "anthropic"], current)
        self.config.llm.kind = kind
        defaults = {
            "auto": None,
            "mock": None,
            "ollama": "http://localhost:11434",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
        }
        if self.config.llm.base_url is None or not str(self.config.llm.base_url).strip():
            self.config.llm.base_url = defaults[kind]
        self.config.llm.model = _prompt("Model", self.config.llm.model or "llama3.2")
        base_default = self.config.llm.base_url or defaults[kind] or ""
        self.config.llm.base_url = _prompt("Base URL", base_default)
        api_default = self.config.llm.api_key or ""
        api_value = _prompt("API key (blank to keep/unset)", api_default)
        self.config.llm.api_key = api_value if api_value else None
        self.config.llm.timeout = _prompt_int("Timeout seconds", self.config.llm.timeout, 1)
        effort_default = self.config.llm.reasoning_effort or "none"
        effort = _prompt("Reasoning effort (none/low/medium/high)", effort_default)
        self.config.llm.reasoning_effort = None if effort.lower() in {"", "none", "off", "null"} else effort.lower()
        self.config.llm.stream = _prompt_bool("Stream responses", self.config.llm.stream)
        self.config.llm.max_output_tokens = _prompt_int("Max output tokens", self.config.llm.max_output_tokens, 64)
        try:
            self.config.validate()
            print("Settings validated.")
        except Exception as exc:  # noqa: BLE001
            print(f"Validation error: {exc}")
        _prompt("Press Enter to continue", "")

    def context_settings(self) -> None:
        _clear()
        print("Context and Agent Settings")
        print(_line())
        self.config.max_tokens = _prompt_int("Max context tokens", self.config.max_tokens, 1024)
        self.config.chars_per_token = _prompt_int("Chars per token", self.config.chars_per_token, 1)
        margin = _prompt("Budget margin (0-0.49)", str(self.config.budget_margin))
        try:
            self.config.budget_margin = float(margin)
        except (TypeError, ValueError):
            pass
        self.config.max_steps = _prompt_int("Max agent steps", self.config.max_steps, 1)
        self.config.keep_recent_messages = _prompt_int("Messages kept during compaction", self.config.keep_recent_messages, 2)
        self.config.subagent_max_tokens = _prompt_int("Subagent context tokens", self.config.subagent_max_tokens, 1024)
        self.config.subagent_max_steps = _prompt_int("Subagent max steps", self.config.subagent_max_steps, 1)
        self.config.max_depth = _prompt_int("Subagent nesting depth", self.config.max_depth, 0)
        self.config.compact_with_llm = _prompt_bool("Summarize old context with the model", self.config.compact_with_llm)
        try:
            self.config.validate()
            print("Settings validated.")
        except Exception as exc:  # noqa: BLE001
            print(f"Validation error: {exc}")
        _prompt("Press Enter to continue", "")

    def execution_settings(self) -> None:
        _clear()
        print("Execution and Security Settings")
        print(_line())
        self.config.workers = _prompt_int("Batch workers", self.config.workers, 1)
        self.config.timeout_shell = _prompt_int("Shell timeout seconds", self.config.timeout_shell, 1)
        self.config.max_shell_output = _prompt_int("Max shell output chars", self.config.max_shell_output, 256)
        self.config.tool_output_limit = _prompt_int("Tool result limit chars", self.config.tool_output_limit, 256)
        self.config.use_shell = _prompt_bool("Use shell mode", self.config.use_shell)
        self.config.offload_shell_output = _prompt_bool("Offload large output to .filament/runs", self.config.offload_shell_output)
        self.config.atomic_writes = _prompt_bool("Atomic file writes", self.config.atomic_writes)
        self.config.allow_outside_target = _prompt_bool("Allow paths outside the workspace", self.config.allow_outside_target)
        self.allow_outside_target = self.config.allow_outside_target
        allowlist = _prompt("Command allowlist (comma separated, blank for none)", ", ".join(self.config.command_allowlist))
        self.config.command_allowlist = [item.strip() for item in allowlist.split(",") if item.strip()]
        try:
            self.config.validate()
            print("Settings validated.")
        except Exception as exc:  # noqa: BLE001
            print(f"Validation error: {exc}")
        _prompt("Press Enter to continue", "")

    def path_settings(self) -> None:
        _clear()
        print("Paths and Defaults")
        print(_line())
        self.shell_cwd = _prompt("Workspace directory", self.shell_cwd or ".")
        self.target = _prompt("Default batch target file", self.target)
        _prompt("Press Enter to continue", "")

    def run_tasks(self) -> None:
        from .orchestrator import Orchestrator

        _clear()
        print("Run Batch Tasks")
        print(_line())
        if not self.tasks:
            task = _prompt("Enter first task (or blank to cancel)")
            if not task:
                return
            self.tasks.append(task)
        target = _prompt("Target file", self.target)
        self.target = target
        try:
            orch = Orchestrator(
                config=self.config,
                shell_cwd=self.shell_cwd,
                allow_outside_target=self.allow_outside_target,
            )
            result = orch.run(
                list(self.tasks),
                target,
                workers=self.config.workers,
                review=self.review,
                fix=self.fix,
                max_fixes=self.max_fixes,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Run failed: {exc}")
            _prompt("Press Enter to continue", "")
            return
        print(_line())
        for key, value in result.items():
            if key == "errors":
                print(f"errors: {len(value)}")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
        _prompt("Press Enter to continue", "")

    def load_config(self) -> None:
        path = _prompt("Config path", str(self.config_path))
        try:
            self.config = load_config(path)
            self.config_path = Path(path)
            self.allow_outside_target = self.config.allow_outside_target
            print("Config loaded.")
        except Exception as exc:  # noqa: BLE001
            print(f"Load failed: {exc}")
        _prompt("Press Enter to continue", "")

    def save_config(self) -> None:
        path = _prompt("Save to", str(self.config_path))
        try:
            _dump_config(Path(path), self.config)
            self.config_path = Path(path)
            print("Config saved.")
        except Exception as exc:  # noqa: BLE001
            print(f"Save failed: {exc}")
        _prompt("Press Enter to continue", "")

    def show_config(self) -> None:
        _clear()
        print("Current Configuration")
        print(_line())
        for key, value in self.config.__dict__.items():
            if key == "llm":
                print("llm:")
                for inner_key, inner_value in self.config.llm.__dict__.items():
                    print(f"  {inner_key}: {inner_value}")
            else:
                print(f"{key}: {value}")
        print("tasks:", self.tasks)
        print("shell_cwd:", self.shell_cwd)
        print("target:", self.target)


def run_tui(config_path: str | Path = "filament.toml") -> None:
    try:
        TUIApp(config_path).main_menu()
    except KeyboardInterrupt:
        print("\nExiting Filament.")
