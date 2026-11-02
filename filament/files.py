"""File helpers for append-mode code integration."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to raw text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    matches = list(re.finditer(r"```(?:[A-Za-z0-9_+-]+)?\r?\n(.*?)```", text, re.DOTALL))
    if matches:
        code = matches[0].group(1).rstrip()
    else:
        # Also support a bare fence without a language/newline-ish odd cases.
        code = text.strip()
    if not code:
        raise ValueError("no code could be extracted")
    return code + "\n"


def append_code(path: str | Path, code: str) -> None:
    if "\x00" in str(path):
        raise ValueError("path contains a null byte")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = code if code.endswith("\n") else code + "\n"
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(normalized)
        f.flush()
        os.fsync(f.fileno())


def replace_line(path: str | Path, line_no: int, new_text: str) -> None:
    """Replace an existing 1-indexed line with new_text atomically."""
    if "\x00" in str(path):
        raise ValueError("path contains a null byte")
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    if line_no < 1 or line_no > len(lines):
        raise IndexError(f"line {line_no} out of range (1..{len(lines)})")
    new_text = new_text.rstrip("\n")
    newline = "\n" if (not lines or lines[line_no - 1].endswith("\n")) else ""
    lines[line_no - 1] = new_text + newline

    fd, temp_name = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(lines))
            f.flush()
            os.fsync(f.fileno())
        backup = p.with_suffix(p.suffix + ".bak")
        if p.exists():
            os.replace(p, backup)
        os.replace(Path(temp_name), p)
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        leftover = Path(temp_name)
        if leftover.exists():
            leftover.unlink(missing_ok=True)


def ensure_target_within_root(target: str | Path, root: str | Path, allow_outside: bool = False) -> Path:
    if "\x00" in str(target) or "\x00" in str(root):
        raise ValueError("path contains a null byte")
    target_path = Path(target).expanduser()
    if not target_path.is_absolute():
        target_path = Path(root).expanduser() / target_path
    target_path = target_path.resolve()
    root_path = Path(root).expanduser().resolve()
    if not allow_outside and not target_path.is_relative_to(root_path):
        raise ValueError(f"target path escapes allowed root: {target_path}")
    return target_path
