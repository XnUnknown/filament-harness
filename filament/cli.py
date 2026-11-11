"""Command line entry point.

``filament`` with no subcommand starts the interactive agent.  ``filament run``
keeps the old batch-task runner for scripted jobs, ``filament tui`` opens the
terminal menu, and ``filament serve`` runs the JSON backend for the Electron UI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .chat import ChatOrchestrator
from .config import HarnessConfig, load_config
from .console import AgentConsole, ConsoleRenderer


def _load_tasks(path: Path) -> tuple[list, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw, "output.py"
    if not isinstance(raw, dict):
        raise ValueError("tasks file must contain a list or an object")
    tasks = raw.get("tasks", [])
    target = raw.get("target", "output.py")
    if tasks is None:
        tasks = []
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty string")
    return tasks, target.strip()


def _apply_overrides(config: HarnessConfig, args: argparse.Namespace) -> HarnessConfig:
    if getattr(args, "client", None):
        config.llm.kind = args.client
    if getattr(args, "model", None):
        config.llm.model = args.model
    if getattr(args, "max_tokens", None):
        config.max_tokens = args.max_tokens
    if getattr(args, "max_steps", None):
        config.max_steps = args.max_steps
    if getattr(args, "reasoning_effort", None):
        config.llm.reasoning_effort = args.reasoning_effort
    if getattr(args, "no_stream", False):
        config.llm.stream = False
    config.validate()
    return config


def _print_banner(chat: ChatOrchestrator) -> None:
    renderer = ConsoleRenderer()
    mode = chat.mode
    provider = f"{chat.config.llm.kind}/{chat.config.llm.model}"
    renderer._line(renderer._paint("filament", "33") + f"  {provider}  ·  mode={mode}")
    renderer._line(renderer._paint(f"workspace {chat.shell_cwd}", "2"))
    renderer._line(renderer._paint("commands: /new /sessions /load /mode /compress /todos /context /help /quit", "2"))


def _handle_command(chat: ChatOrchestrator, command: str) -> bool:
    parts = command.strip().split()
    name = parts[0].lower()
    if name in {"/quit", "/exit", "exit", "quit"}:
        return False
    if name == "/help":
        print("Session: /new, /sessions, /load <id>, /mode build|plan, /compress, /todos, /context, /quit")
        return True
    if name == "/new":
        session = chat.new_session(parts[1] if len(parts) > 1 else chat.mode)
        print(f"new session: {session.session_id}")
        return True
    if name == "/sessions":
        sessions = ChatOrchestrator.list_sessions(chat.shell_cwd)
        if not sessions:
            print("(no saved sessions)")
        for session in sessions[:20]:
            print(f"{session.session_id}  {session.display_title[:60]}  ({session.status})")
        return True
    if name == "/load":
        if len(parts) != 2:
            print("usage: /load <session_id>")
            return True
        try:
            chat.load_session(parts[1])
            print(f"loaded {parts[1]} ({len(chat.messages)} messages)")
        except Exception as exc:  # noqa: BLE001
            print(f"load failed: {exc}")
        return True
    if name == "/mode":
        if len(parts) != 2 or parts[1].lower() not in {"build", "plan"}:
            print("usage: /mode build|plan")
            return True
        print(f"mode: {chat.set_mode(parts[1])}")
        return True
    if name == "/compress":
        stats = chat.compress()
        print(f"tokens {stats.get('before_tokens')} -> {stats.get('after_tokens')}")
        return True
    if name == "/todos":
        todos = chat.session.todos
        if not todos:
            print("(no todos)")
        for item in todos:
            mark = {"completed": "[x]", "in_progress": "[~]"}.get(str(item.get("status")), "[ ]")
            print(f"{mark} {item.get('content', '')}")
        return True
    if name == "/context":
        state = chat.state()
        print(f"{state['tokens']}/{state['limit']} tokens · {state['messages']} messages · {state['status']}")
        return True
    print(f"unknown command: {name} (try /help)")
    return True


def _chat_repl(config: HarnessConfig, args: argparse.Namespace) -> int:
    chat = ChatOrchestrator(config, getattr(args, "shell_cwd", "."), mode=getattr(args, "mode", "build"))
    console = AgentConsole(chat)
    _print_banner(chat)
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
            chat.cancel()
            print("\n[cancelling after the current model call]")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}")
            continue
        console.print_result(result)
    chat.save()
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    from .orchestrator import Orchestrator

    try:
        config = _apply_overrides(load_config(args.config), args)
    except Exception as exc:  # noqa: BLE001
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    tasks_path = Path(args.tasks_file) if args.tasks_file else None
    if tasks_path is None:
        print("tasks file is required", file=sys.stderr)
        return 2

    try:
        tasks, file_target = _load_tasks(tasks_path)
        target = args.target or file_target
        orch = Orchestrator(
            config=config,
            shell_cwd=args.shell_cwd,
            allow_outside_target=args.allow_outside_target,
        )
        if not args.quiet:
            print(f"running {len(tasks)} task(s) -> {target}", file=sys.stderr)
        result = orch.run(
            tasks,
            target,
            workers=args.workers,
            review=not args.no_review,
            fix=not args.no_fix,
            max_fixes=args.max_fixes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"run error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(result, indent=2)
    print(output)
    if args.report:
        Path(args.report).write_text(output + "\n", encoding="utf-8")
    return 1 if result.get("errors") else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="filament", description="Lightweight agentic coding harness")
    parser.add_argument("--version", action="version", version=f"filament {__version__}")
    # Global chat options so `filament --client mock` works without a subcommand.
    # SUPPRESS keeps them absent unless the user actually passes them, which lets
    # the `chat` subparser supply its own defaults without conflicts.
    parser.add_argument("--config", default=argparse.SUPPRESS, help="config path")
    parser.add_argument("--shell-cwd", default=argparse.SUPPRESS, help="workspace directory")
    parser.add_argument("--mode", choices=["build", "plan"], default=argparse.SUPPRESS)
    parser.add_argument("--client", choices=["auto", "mock", "ollama", "openai", "anthropic"], default=argparse.SUPPRESS)
    parser.add_argument("--model", default=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=argparse.SUPPRESS)
    parser.add_argument("--no-stream", action="store_true", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="cmd")

    chat = sub.add_parser("chat", help="interactive agent (default)")
    chat.add_argument("--config", default=argparse.SUPPRESS, help="config path")
    chat.add_argument("--shell-cwd", default=argparse.SUPPRESS, help="workspace directory")
    chat.add_argument("--mode", choices=["build", "plan"], default=argparse.SUPPRESS)
    chat.add_argument("--client", choices=["auto", "mock", "ollama", "openai", "anthropic"], default=argparse.SUPPRESS)
    chat.add_argument("--model", default=argparse.SUPPRESS)
    chat.add_argument("--max-tokens", type=int, default=argparse.SUPPRESS)
    chat.add_argument("--max-steps", type=int, default=argparse.SUPPRESS)
    chat.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=argparse.SUPPRESS)
    chat.add_argument("--no-stream", action="store_true", default=argparse.SUPPRESS)

    run = sub.add_parser("run", help="run a batch of coding tasks (legacy)")
    run.add_argument("tasks_file", nargs="?", help="JSON file with tasks")
    run.add_argument("--config", help="path to TOML config")
    run.add_argument("--target", help="override target file")
    run.add_argument("--client", choices=["auto", "mock", "ollama", "openai", "anthropic"])
    run.add_argument("--model", help="override LLM model")
    run.add_argument("--max-tokens", type=int, help="context token budget")
    run.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)
    run.add_argument("--workers", type=int, help="parallel agent count")
    run.add_argument("--shell-cwd", default=".", help="working directory")
    run.add_argument("--allow-outside-target", action="store_true", help="allow target outside shell cwd")
    run.add_argument("--no-review", action="store_true", help="skip review phase")
    run.add_argument("--no-fix", action="store_true", help="skip line fixes")
    run.add_argument("--max-fixes", type=int, help="maximum review fixes")
    run.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="reasoning effort")
    run.add_argument("--no-stream", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--report", help="write JSON report to this path")
    run.add_argument("--quiet", action="store_true", help="suppress progress")

    tui = sub.add_parser("tui", help="interactive terminal UI")
    tui.add_argument("--config", default="filament.toml", help="config path")

    sub.add_parser("serve", help="run the JSON backend for the Electron app")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = args.cmd or "chat"

    if command == "tui":
        from .tui import run_tui

        run_tui(args.config)
        return 0
    if command == "serve":
        from .backend_server import main as backend_main

        backend_main()
        return 0
    if command == "run":
        return _run_batch(args)

    try:
        config = _apply_overrides(load_config(getattr(args, "config", "filament.toml")), args)
    except Exception as exc:  # noqa: BLE001
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return _chat_repl(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
