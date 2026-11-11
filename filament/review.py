"""Review helpers: find problematic lines without bloating context."""
from __future__ import annotations

import re
from pathlib import Path
from .context import ContextBudget
from .llm import LLMClient


REVIEW_SYSTEM = (
    "You are a code reviewer. List only fixes needed as lines: "
    "'FIX <line_number>: <instruction>'. No other text."
)

_FIX_RE = re.compile(r"(?:^|\n)\s*(?:FIX|Fix|fix)\s*(?:line\s*)?(\d+)\s*[:\-]\s*(.+?)(?=\n|$)", re.IGNORECASE)


def parse_fixes(text: str) -> list[tuple[int, str]]:
    fixes: list[tuple[int, str]] = []
    seen: set[int] = set()
    for match in _FIX_RE.finditer(text):
        line_no = int(match.group(1))
        instruction = match.group(2).strip()
        if line_no > 0 and instruction and line_no not in seen:
            seen.add(line_no)
            fixes.append((line_no, instruction))
    return fixes


def review_file(client: LLMClient, path: str | Path, budget: ContextBudget) -> list[tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Keep review input light: cap to remaining budget with a sensible floor.
    remaining_chars = max(2000, budget.remaining_tokens() * budget.chars_per_token)
    line_offset = 0
    if len(text) > remaining_chars:
        # Trim from the head, but start on a line boundary and remember how many
        # lines were dropped so reported line numbers still refer to the real file.
        cut = len(text) - remaining_chars
        newline = text.find("\n", cut)
        cut = len(text) if newline == -1 else newline + 1
        line_offset = text.count("\n", 0, cut)
        text = text[cut:]
    prompt = f"Review this file and list fixes:\n```\n{text}\n```"
    budget.add(prompt)
    raw = client.generate(prompt, system=REVIEW_SYSTEM)
    budget.add(raw)
    fixes = parse_fixes(raw)
    if line_offset:
        fixes = [(line_no + line_offset, instruction) for line_no, instruction in fixes]
    return fixes
