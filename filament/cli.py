"""CLI entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(prog="filament")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a batch of coding tasks")
    run.add_argument("tasks_file", help="JSON file with {tasks:[...], target:'file.py'}")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--no-review", action="store_true")
    run.add_argument("--no-fix", action="store_true")
    run.add_argument("--shell-cwd", default=".")
    run.add_argument("--client", default="auto", choices=["auto", "mock", "ollama"])

    args = parser.parse_args()
    if args.cmd == "run":
        data = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
        orch = Orchestrator(client_kind=args.client, shell_cwd=args.shell_cwd)
        result = orch.run(
            data.get("tasks", []),
            data.get("target", "output.py"),
            workers=args.workers,
            review=not args.no_review,
            fix=not args.no_fix,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
