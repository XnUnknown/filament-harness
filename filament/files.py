"""File helpers for append-mode code integration."""
from __future__ import annotations

import re
from pathlib import Path


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to raw text."""
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip() + "\n"
    return text.strip() + "\n"


def append_code(path: str | Path, code: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(code)
        if not code.endswith("\n"):
            f.write("\n")


def replace_line(path: str | Path, line_no: int, new_text: str) -> None:
    """Replace an existing 1-indexed line with new_text."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    if line_no < 1 or line_no > len(lines):
        raise IndexError(f"line {line_no} out of range (1..{len(lines)})")
    # Preserve newline style
    newline = "\n" if lines[line_no - 1].endswith("\n") else ""
    lines[line_no - 1] = new_text + ("" if new_text.endswith("\n") else newline)
    p.write_text("".join(lines), encoding="utf-8")
