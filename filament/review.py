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


def review_file(client: LLMClient, path: str | Path, budget: ContextBudget) -> list[tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Keep review input light: cap to budget remainder
    cap_chars = max(4000, budget.max_chars - budget.total_tokens() * 4)
    if len(text) > cap_chars:
        text = text[-cap_chars:]
    prompt = f"Review this file and list fixes:\n```\n{text}\n```"
    budget.add(prompt)
    raw = client.generate(prompt, system=REVIEW_SYSTEM)
    budget.add(raw)
    fixes = []
    for m in re.finditer(r"FIX\s+(\d+)\s*:\s*(.+)", raw):
        fixes.append((int(m.group(1)), m.group(2).strip()))
    return fixes
