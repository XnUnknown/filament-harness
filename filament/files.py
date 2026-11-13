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
    matches = list(re.finditer(r"```[^\r\n]*\r?\n(.*?)```", text, re.DOTALL))
    if matches:
        code = matches[0].group(1).rstrip()
    else:
        # Handle malformed output where a fenced block was opened but not closed
        # (or where text contains a single fence line). Take everything after
        # the first fence opener and strip a trailing fence if one exists.
        stripped = text.strip()
        fence_match = re.search(r"```[^\r\n]*\r?\n", stripped)
        if fence_match is not None:
            code = stripped[fence_match.end():]
            closing = code.rfind("```")
            if closing != -1:
                code = code[:closing]
            code = code.rstrip()
        else:
            code = stripped
    if not code:
        raise ValueError("no code could be extracted")
    return code + "\n"


def _line_ending(line: str) -> str:
    """Return the line ending a single line already carries."""
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def _read_preserving_newlines(path: Path) -> str:
    """Read text without universal-newline translation.

    ``Path.read_text`` normalizes CRLF to LF, so writing that text back turns a
    Windows file into an LF file.  Reading with ``newline=""`` keeps every
    line's own ending, which is what a one-line edit or an append must do.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _file_ending(path: Path) -> str:
    """Return the dominant line ending of an existing file (default LF)."""
    try:
        with path.open("rb") as handle:
            sample = handle.read(65_536)
    except OSError:
        return "\n"
    crlf = sample.count(b"\r\n")
    lf = sample.count(b"\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _in_ending(text: str, ending: str) -> str:
    """Normalize ``text`` to ``ending`` and guarantee one final newline."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if ending != "\n":
        normalized = normalized.replace("\n", ending)
    if not normalized.endswith(ending):
        normalized += ending
    return normalized


def _needs_separator(path: Path) -> bool:
    """True when the file's last byte is not a line ending."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) not in (b"\n", b"\r")
    except OSError:
        return False


def append_code(
    path: str | Path,
    code: str,
    atomic: bool = False,
    *,
    ensure_line_start: bool = True,
) -> None:
    """Append code to path; optionally rewrite the file atomically.

    The existing file's line endings are preserved: appending LF lines to a
    CRLF file turns a generated block into a mixed-ending file.  When
    ``ensure_line_start`` is true (the code-append case) a missing final
    newline on the existing file is repaired first, or the new block would be
    glued onto the last line.  A plain text append can disable that to extend
    the current line deliberately.
    """
    if "\x00" in str(path):
        raise ValueError("path contains a null byte")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(path)):
        raise ValueError("path contains a control character")
    p = Path(path)
    if p.drive and not p.root:
        raise ValueError("path contains a drive-relative component")
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    ending = _file_ending(p) if existed else "\n"
    block = _in_ending(code, ending)
    if ensure_line_start and existed and _needs_separator(p):
        block = ending + block

    if not atomic:
        with p.open("ab") as f:
            f.write(block.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        return

    existing = _read_preserving_newlines(p) if existed else ""
    combined = existing + block

    fd, temp_name = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(combined)
            f.flush()
            os.fsync(f.fileno())
        if p.exists():
            try:
                os.chmod(temp_name, p.stat().st_mode)
            except OSError:
                pass
        os.replace(Path(temp_name), p)
    finally:
        leftover = Path(temp_name)
        if leftover.exists():
            leftover.unlink(missing_ok=True)


def replace_line(path: str | Path, line_no: int, new_text: str) -> None:
    """Replace an existing 1-indexed line with new_text atomically.

    Only that line changes.  Its own line ending is kept, so replacing one line
    in a CRLF file does not rewrite every other line as LF.
    """
    if "\x00" in str(path):
        raise ValueError("path contains a null byte")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(path)):
        raise ValueError("path contains a control character")
    p = Path(path)
    if p.drive and not p.root:
        raise ValueError("path contains a drive-relative component")
    raw = _read_preserving_newlines(p)
    lines = raw.splitlines(keepends=True)
    if line_no < 1 or line_no > len(lines):
        raise IndexError(f"line {line_no} out of range (1..{len(lines)})")
    ending = _line_ending(lines[line_no - 1])
    lines[line_no - 1] = new_text.rstrip("\r\n") + ending

    backup = p.with_suffix(p.suffix + ".bak")
    fd, temp_name = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write("".join(lines))
            f.flush()
            os.fsync(f.fileno())
        if p.exists():
            try:
                os.chmod(temp_name, p.stat().st_mode)
            except OSError:
                pass
            os.replace(p, backup)
        os.replace(Path(temp_name), p)
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            pass
    except Exception:
        # Restore the original file if the replacement failed midway.
        if Path(temp_name).exists():
            Path(temp_name).unlink(missing_ok=True)
        if not p.exists() and backup.exists():
            try:
                os.replace(backup, p)
            except OSError:
                pass
        raise
    finally:
        leftover = Path(temp_name)
        if leftover.exists():
            leftover.unlink(missing_ok=True)


def ensure_target_within_root(target: str | Path, root: str | Path, allow_outside: bool = False) -> Path:
    if "\x00" in str(target) or "\x00" in str(root):
        raise ValueError("path contains a null byte")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(target) + str(root)):
        raise ValueError("path contains a control character")
    target_path = Path(target).expanduser()
    if target_path.drive and not target_path.root:
        raise ValueError("path contains a drive-relative component")
    if not target_path.is_absolute():
        target_path = Path(root).expanduser() / target_path
    target_path = target_path.resolve()
    root_path = Path(root).expanduser().resolve()
    if not allow_outside and not target_path.is_relative_to(root_path):
        raise ValueError(f"target path escapes allowed root: {target_path}")
    return target_path
