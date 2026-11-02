"""Interactive terminal user interface for Filament Harness.

The TUI uses only the Python standard library so it works in the same minimal
environment as the harness itself.
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import HarnessConfig, load_config
from .llm import build_client
from .orchestrator import Orchestrator


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
    allow = "[" + ", ".join(repr(item) for item in config.command_allowlist) + "]"
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
        f"command_allowlist = {allow}",
        "",
        "[llm]",
        f'kind = "{llm.kind}"',
        f'model = "{llm.model}"',
        f'base_url = {repr(llm.base_url or "")}',
        f'api_key = {repr(llm.api_key or "")}',
        f"timeout = {llm.timeout}",
    ]
    if llm.reasoning_effort:
        lines.insert(-1, f'reasoning_effort = "{llm.reasoning_effort}"')
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

    def main_menu(self) -> None:
        while True:
            _clear()
            print("Filament Harness TUI")
            print(_line())
            print(f"Config: {self.config_path} | Provider: {self.config.llm.kind} | Model: {self.config.llm.model}")
            print(f"Tasks: {len(self.tasks)} | Target: {self.target} | CWD: {self.shell_cwd}")
            print(_line())
            print("1. Run tasks")
            print("c. Chat with orchestrator")
            print("2. Manage tasks")
            print("3. Provider settings")
            print("4. Context and budget settings")
            print("5. Execution and security settings")
            print("6. Paths and task defaults")
            print("7. Run settings")
            print("8. Load config")
            print("9. Save config")
            print("0. Show current config")
            print("q. Quit")
            choice = _prompt("Select", "1")
            if choice == "1":
                self.run_tasks()
            elif choice.lower() in {"c", "chat"}:
                self.chat_mode()
            elif choice == "2":
                self.manage_tasks()
            elif choice == "3":
                self.provider_settings()
            elif choice == "4":
                self.context_settings()
            elif choice == "5":
                self.execution_settings()
            elif choice == "6":
                self.path_settings()
            elif choice == "7":
                self.run_settings()
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

    def manage_tasks(self) -> None:
        while True:
            _clear()
            print("Tasks")
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
        kind = _prompt_choice("Provider", ["mock", "ollama", "openai", "anthropic"], current)
        self.config.llm.kind = kind
        defaults = {
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
        try:
            self.config.validate()
            print("Settings validated.")
        except Exception as exc:  # noqa: BLE001
            print(f"Validation error: {exc}")
        _prompt("Press Enter to continue", "")

    def context_settings(self) -> None:
        _clear()
        print("Context and Budget Settings")
        print(_line())
        self.config.max_tokens = _prompt_int("Max tokens", self.config.max_tokens, 1024)
        self.config.chars_per_token = _prompt_int("Chars per token", self.config.chars_per_token, 1)
        margin = _prompt("Budget margin (0-0.49)", str(self.config.budget_margin))
        try:
            self.config.budget_margin = float(margin)
        except (TypeError, ValueError):
            pass
        self.config.max_tasks = _prompt_int("Max tasks", self.config.max_tasks, 1)
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
        self.config.workers = _prompt_int("Workers", self.config.workers, 1)
        self.config.timeout_shell = _prompt_int("Shell timeout seconds", self.config.timeout_shell, 1)
        self.config.max_shell_output = _prompt_int("Max shell output chars", self.config.max_shell_output, 256)
        self.config.use_shell = _prompt_bool("Use shell mode", self.config.use_shell)
        self.config.allow_outside_target = _prompt_bool("Allow targets outside working root", self.config.allow_outside_target)
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
        print("Paths and Task Defaults")
        print(_line())
        self.shell_cwd = _prompt("Shell working directory", self.shell_cwd or ".")
        self.target = _prompt("Default target file", self.target)
        print(f"Client builder: {build_client(self.config.llm).__class__.__name__}")
        _prompt("Press Enter to continue", "")

    def run_settings(self) -> None:
        _clear()
        print("Run Settings")
        print(_line())
        self.review = _prompt_bool("Enable review", self.review)
        self.fix = _prompt_bool("Enable line fixes", self.fix)
        max_fixes = _prompt("Max fixes (blank for all)", "" if self.max_fixes is None else str(self.max_fixes))
        self.max_fixes = int(max_fixes) if max_fixes.strip() else None
        _prompt("Press Enter to continue", "")

    def chat_mode(self) -> None:
        from filament.chat import ChatOrchestrator

        chat = ChatOrchestrator(self.config, self.shell_cwd, mode=self._chat_mode_name())
        print("Chat with main orchestrator")
        print(_line())
        print(f"Session: {chat.session.session_id} | Mode: {chat.mode} | Provider: {self.config.llm.kind} | Model: {self.config.llm.model}")
        print("If provider shows mock, run this TUI from the project directory containing .env, or pass --config.")
        print("Commands: /mode auto|plan|review|accept, /sessions, /load <id>, /compress, /config, exit")
        print("Type your request, or 'exit' to return.")
        while True:
            request = _prompt("You")
            if not request or request.lower() in {"exit", "quit", "q"}:
                chat.save()
                return
            try:
                if request.startswith("/"):
                    self._handle_chat_command(chat, request)
                    continue
                if chat.is_small_talk(request):
                    print("Orchestrator:", chat.direct_chat(request))
                    continue
                summary, tasks = chat.plan(request)
                print(f"Orchestrator: {summary}")
                print("Planned tasks:")
                for task in tasks:
                    print(f"  - [{task['id']}] {task['task']} -> {task['target']}")
                should_run = chat.mode == "auto" or _prompt_bool("Run this plan now?", True)
                if not should_run:
                    continue
                result = chat.run_plan(tasks, review=self.review, fix=self.fix, max_fixes=self.max_fixes)
                print(_line())
                for key, value in result.items():
                    if key == "errors":
                        print(f"errors: {len(value)}")
                        for item in value:
                            print(f"  - {item}")
                    elif key != "plan_summary":
                        print(f"{key}: {value}")
                target = Path(result.get("target_file", "output.py"))
                if target.exists() and target.stat().st_size <= 20000:
                    print(_line())
                    print(target.read_text(encoding="utf-8", errors="replace"))
                print(f"Session: {chat.session.session_id} | Mode: {chat.mode}")
                # Autonomous review/follow-up loop: continue until orchestrator says done or no more plan.
                for _ in range(5):
                    follow_up = chat.reflect(result)
                    if not follow_up:
                        break
                    follow_summary, follow_tasks = follow_up
                    print(f"Orchestrator follow-up: {follow_summary}")
                    print("Planned tasks:")
                    for task in follow_tasks:
                        print(f"  - [{task['id']}] {task['task']} -> {task['target']}")
                    if chat.mode != "auto" and not _prompt_bool("Run follow-up plan?", True):
                        break
                    follow_result = chat.run_plan(follow_tasks, review=self.review, fix=self.fix, max_fixes=self.max_fixes)
                    print(_line())
                    for key, value in follow_result.items():
                        if key == "errors":
                            print(f"errors: {len(value)}")
                            for item in value:
                                print(f"  - {item}")
                        elif key != "plan_summary":
                            print(f"{key}: {value}")
                    result = follow_result
                    target = Path(result.get("target_file", "output.py"))
                    if target.exists() and target.stat().st_size <= 20000:
                        print(_line())
                        print(target.read_text(encoding="utf-8", errors="replace"))
                print(f"Session: {chat.session.session_id} | Mode: {chat.mode}")
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
            # Let the orchestrator loop continue immediately; session is saved after each plan.

    def _chat_mode_name(self) -> str:
        return "auto"

    def _handle_chat_command(self, chat, command: str) -> None:
        from filament.chat import ChatOrchestrator  # noqa: F401

        parts = command.strip().split()
        name = parts[0].lower()
        if name == "/mode":
            if len(parts) != 2 or parts[1].lower() not in {"auto", "plan", "review", "accept"}:
                print("Usage: /mode auto|plan|review|accept")
                return
            chat.mode = parts[1].lower()
            chat.session.mode = chat.mode
            chat.save()
            print(f"Mode set to {chat.mode}.")
            return
        if name == "/sessions":
            sessions = ChatOrchestrator.list_sessions(self.shell_cwd)
            if not sessions:
                print("(no saved sessions)")
                return
            for session in sessions[:20]:
                print(f"{session.session_id}  mode={session.mode}  updated={session.updated_at:.0f}")
            return
        if name == "/load":
            if len(parts) != 2:
                print("Usage: /load <session_id>")
                return
            try:
                chat.load_session(parts[1])
                print(f"Loaded session {parts[1]} (mode={chat.mode}).")
            except Exception as exc:  # noqa: BLE001
                print(f"Load failed: {exc}")
            return
        if name == "/compress":
            changed = chat.compress_if_needed()
            print("Context compressed." if changed else "No compression needed.")
            return
        if name == "/config":
            print(f"cwd={self.shell_cwd} mode={chat.mode} session={chat.session.session_id}")
            return
        print("Unknown command. Commands: /mode, /sessions, /load, /compress, /config")

    def run_tasks(self) -> None:
        _clear()
        print("Run Tasks")
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
        print("Result:")
        for key, value in result.items():
            if key == "errors":
                print(f"errors: {len(value)}")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
        target_path = Path(result.get("target_file", target))
        if target_path.exists() and target_path.stat().st_size <= 20000:
            print(_line())
            print("Generated file:")
            print(target_path.read_text(encoding="utf-8", errors="replace"))
        _prompt("Press Enter to continue", "")

    def load_config(self) -> None:
        path = _prompt("Config path", str(self.config_path))
        try:
            new_config = load_config(path)
            self.config = new_config
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
        print("review:", self.review, "fix:", self.fix, "max_fixes:", self.max_fixes)


def run_tui(config_path: str | Path = "filament.toml") -> None:
    try:
        TUIApp(config_path).main_menu()
    except KeyboardInterrupt:
        print("\nExiting Filament TUI.")
