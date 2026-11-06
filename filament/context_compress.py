"""Context compression for long Filament chat sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .context import ContextBudget
from .llm import LLMClient


def estimate_messages(messages: list[dict[str, Any]], chars_per_token: int = 4) -> int:
    total = 0
    for message in messages:
        content = message.get("content", "")
        total += max(1, len(str(content)) // chars_per_token)
    return total


def compress_messages(
    client: LLMClient,
    messages: list[dict[str, Any]],
    max_tokens: int,
    chars_per_token: int = 4,
    keep_last: int = 6,
    threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], bool]:
    """Compress older messages into one summary when near the token budget."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if chars_per_token < 1:
        raise ValueError("chars_per_token must be positive")
    if keep_last < 1:
        keep_last = 1
    if not 0.0 < threshold < 1.0:
        threshold = min(max(threshold, 0.01), 0.99)
    if len(messages) <= keep_last:
        return messages, False
    if estimate_messages(messages, chars_per_token) <= int(max_tokens * threshold):
        return messages, False

    older = messages[:-keep_last]
    recent = messages[-keep_last:]
    transcript = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in older)
    # Keep the summarization request itself inside the configured budget, with a
    # generous response margin, instead of forwarding unbounded conversation text.
    instruction = (
        "Summarize the following conversation for an ongoing coding agent. "
        "Preserve user goals, decisions, changed files, errors, and open tasks. "
        "Be concise.\n\n"
    )
    truncation_marker = "\n... [earlier context truncated] ..."
    # Reserve space for the instruction and a truncation marker so the generated
    # prompt cannot exceed the configured character budget.
    max_transcript_chars = max(
        32,
        max_tokens * chars_per_token - len(instruction) - len(truncation_marker) - 64,
    )
    if len(transcript) > max_transcript_chars:
        transcript = transcript[:max_transcript_chars] + truncation_marker
    prompt = instruction + transcript
    try:
        summary = client.generate(prompt, system="Return only a concise summary.")
    except Exception:  # noqa: BLE001
        # Fall back to a deterministic truncation if the model fails.
        summary = transcript[: max(200, max_tokens * chars_per_token // 2)]
    compressed = [{"role": "assistant", "content": f"CONTEXT SUMMARY: {summary}"}] + recent
    # Avoid an impossible resulting state; keep recent if summary itself is too large.
    if estimate_messages(compressed, chars_per_token) > max_tokens:
        return messages[-keep_last:], True
    return compressed, True


def budget_for_model(max_tokens: int, margin: float = 0.05) -> ContextBudget:
    return ContextBudget(max_tokens=max(1024, int(max_tokens * (1 - margin))), chars_per_token=4)
