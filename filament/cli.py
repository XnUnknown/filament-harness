"""CLI entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .orchestrator import Orchestrator


def _load_tasks(path: Path) -> tuple[list, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw, "output.py"
    if not isinstance(raw, dict):
        raise ValueError("tasks file must contain a list or an object")
    tasks = raw.get("tasks", [])
    target = raw.get("target", "output.py")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty string")
    return tasks, target.strip()


def main() -> int:
    parser = argparse.ArgumentParser(prog="filament", description="Lightweight local LLM coding harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a batch of coding tasks")
    run.add_argument("tasks_file", nargs="?", help="JSON file with tasks")
    run.add_argument("--config", help="path to TOML config")
    run.add_argument("--target", help="override target file")
    run.add_argument("--client", choices=["auto", "mock", "ollama", "openai", "anthropic"])
    run.add_argument("--model", help="override LLM model")
    run.add_argument("--max-tokens", type=int, help="context token budget")
    run.add_argument("--workers", type=int, help="parallel agent count")
    run.add_argument("--shell-cwd", default=".", help="working directory")
    run.add_argument("--allow-outside-target", action="store_true", help="allow target outside shell cwd")
    run.add_argument("--no-review", action="store_true", help="skip review phase")
    run.add_argument("--no-fix", action="store_true", help="skip line fixes")
    run.add_argument("--max-fixes", type=int, help="maximum review fixes")
    run.add_argument("--report", help="write JSON report to this path")
    run.add_argument("--quiet", action="store_true", help="suppress progress")

    args = parser.parse_args()

    if args.cmd != "run":
        return 2

    try:
        config = load_config(args.config)
        if args.client is not None:
            config.llm.kind = args.client
        if args.model is not None:
            config.llm.model = args.model
        if args.max_tokens is not None:
            config.max_tokens = args.max_tokens
        if args.workers is not None:
            config.workers = args.workers
        config.validate()
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


if __name__ == "__main__":
    raise SystemExit(main())
