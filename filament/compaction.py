"""Context compaction for long agent sessions.

A harness lives or dies by what stays in the prompt.  The rules here are
deliberate:

* tool results are the cheapest thing to shrink, so they are pruned first;
* conversation history is cut only at a user turn, never between an assistant
  tool call and its results (that would make the next request invalid);
* the removed turns are replaced by one dense digest, optionally rewritten by
  the model, so the agent keeps goals and outcomes without the transcript.
"""
from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient

SUMMARY_MARKER = "[context summary]"
PROMPT_PREFIX = (
    SUMMARY_MARKER
    + " Earlier conversation was compacted to keep the context window small. "
    "This summary replaces it.\n"
)


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def message_tokens(message: dict[str, Any], chars_per_token: int = 4) -> int:
    total = len(_content_text(message))
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            total += len(str(call.get("name", ""))) + len(
                json.dumps(call.get("arguments") or {}, ensure_ascii=False)
            )
    return max(1, total // max(1, chars_per_token))


def estimate_tokens(messages: list[dict[str, Any]], chars_per_token: int = 4) -> int:
    return sum(message_tokens(message, chars_per_token) for message in messages)


def prune_tool_results(
    messages: list[dict[str, Any]],
    keep_recent: int = 8,
    limit_chars: int = 1200,
) -> tuple[list[dict[str, Any]], int]:
    """Replace large tool results outside the recent window with a marker."""
    if keep_recent < 0:
        keep_recent = 0
    boundary = max(0, len(messages) - keep_recent)
    pruned = 0
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if (
            index < boundary
            and message.get("role") == "tool"
            and len(_content_text(message)) > limit_chars
        ):
            original = len(_content_text(message))
            shortened = dict(message)
            shortened["content"] = (
                f"[tool output omitted during compaction: {original} chars]"
            )
            result.append(shortened)
            pruned += 1
        else:
            result.append(message)
    return result, pruned


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def build_digest(messages: list[dict[str, Any]], limit_chars: int = 6000) -> str:
    """Deterministic summary used directly or as input to the summarizer."""
    goals: list[str] = []
    assistant_notes: list[str] = []
    commands: list[str] = []
    files: list[str] = []
    outcomes: list[str] = []

    for message in messages:
        role = message.get("role")
        text = _content_text(message).strip()
        if role == "user" and text:
            goals.append(_short(text, 240))
        elif role == "assistant":
            if text:
                assistant_notes.append(_short(text, 300))
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "")
                args = call.get("arguments") or {}
                if name == "run_shell" and isinstance(args, dict):
                    commands.append(_short(str(args.get("command", "")), 160))
                elif name in {"write_file", "edit_file", "multi_edit", "read_file"} and isinstance(args, dict):
                    path = args.get("path")
                    if path and str(path) not in files:
                        files.append(str(path))
        elif role == "tool" and text:
            outcomes.append(_short(text, 140))

    lines: list[str] = []
    if goals:
        lines.append("User requests:")
        lines.extend(f"- {item}" for item in goals[-8:])
    if files:
        lines.append("Files touched:")
        lines.extend(f"- {item}" for item in files[:16])
    if commands:
        lines.append("Commands run:")
        lines.extend(f"- {item}" for item in commands[-8:])
    if assistant_notes:
        lines.append("Assistant notes:")
        lines.extend(f"- {item}" for item in assistant_notes[-5:])
    if outcomes:
        lines.append("Recent tool outcomes:")
        lines.extend(f"- {item}" for item in outcomes[-10:])
    digest = "\n".join(lines) if lines else "(no usable history)"
    return digest[:limit_chars]


def _summarize(
    client: LLMClient | None,
    older: list[dict[str, Any]],
    max_chars: int,
    use_llm: bool,
) -> tuple[str, bool]:
    digest = build_digest(older, limit_chars=min(6000, max_chars * 2))
    if not use_llm or client is None or len(digest) > max_chars:
        return digest[:max_chars], False
    prompt = (
        "Summarize this coding session for an agent that must continue it. "
        "Preserve: user goals and constraints, decisions made, files created or "
        "changed, commands and their outcomes, errors still open, and the next "
        "step. Be terse and factual. No preamble.\n\n"
        + digest
    )
    try:
        summary = client.generate(prompt, system="Return only a dense factual summary.")
    except Exception:  # noqa: BLE001 - a failed summary must never abort a run
        return digest[:max_chars], False
    summary = str(summary or "").strip()
    if not summary:
        return digest[:max_chars], False
    return summary[:max_chars], True


def _safe_cut(conversation: list[dict[str, Any]], keep_recent: int) -> int:
    """Find an index the recent window can start at without breaking pairing."""
    if keep_recent < 1:
        keep_recent = 1
    target = max(0, len(conversation) - keep_recent)
    if target <= 0:
        return 0
    # Prefer a user turn: the window then starts at a fresh request.
    cut = target
    while cut > 0 and conversation[cut].get("role") != "user":
        cut -= 1
    if cut > 0:
        return cut
    # A single long-running turn has no later user boundary.  Cut at the first
    # non-tool message instead: an assistant message keeps its own tool results
    # in the window, and advancing past tool results never leaves an orphan.
    cut = target
    while cut < len(conversation) and conversation[cut].get("role") == "tool":
        cut += 1
    if cut < len(conversation):
        return cut
    # The whole tail is one tool batch (the common shape after spawn_agents or
    # a wide read/grep fan-out).  Advancing past its results reached the end,
    # but giving up here left the prompt over budget: start the window at the
    # assistant turn that issued the batch, which keeps every result with its
    # call and still lets the older turns be summarized.
    cut = len(conversation) - 1
    while cut > 0 and conversation[cut].get("role") == "tool":
        cut -= 1
    return cut if cut > 0 else 0


def _drop_oldest_step(recent: list[dict[str, Any]]) -> None:
    """Drop one pairing-safe chunk from the start of the recent window."""
    if not recent:
        return
    role = recent[0].get("role")
    recent.pop(0)
    if role == "user":
        # Prefer dropping a whole old turn, but keep the work that follows when
        # this is the only user message left in the window.
        if any(message.get("role") == "user" for message in recent):
            while recent and recent[0].get("role") != "user":
                recent.pop(0)
        else:
            while recent and recent[0].get("role") == "tool":
                recent.pop(0)
        return
    # An assistant message owns the tool results that follow it.
    while recent and recent[0].get("role") == "tool":
        recent.pop(0)


def compact_messages(
    client: LLMClient | None,
    messages: list[dict[str, Any]],
    max_tokens: int,
    chars_per_token: int = 4,
    keep_recent: int = 8,
    threshold: float = 0.75,
    use_llm: bool = True,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Compact ``messages`` in place-free fashion; returns (messages, changed, stats)."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if chars_per_token < 1:
        raise ValueError("chars_per_token must be positive")
    target = max(1, int(max_tokens * max(0.1, min(threshold, 0.99))))
    stats: dict[str, Any] = {
        "before_tokens": estimate_tokens(messages, chars_per_token),
        "after_tokens": 0,
        "pruned_results": 0,
        "summarized_messages": 0,
        "used_llm": False,
    }
    if stats["before_tokens"] <= target:
        stats["after_tokens"] = stats["before_tokens"]
        return messages, False, stats

    working, pruned = prune_tool_results(messages, keep_recent=keep_recent)
    stats["pruned_results"] = pruned

    systems = [m for m in working if m.get("role") == "system"]
    conversation = [m for m in working if m.get("role") != "system"]
    cut = _safe_cut(conversation, keep_recent)
    if cut <= 0:
        # No turn boundary exists (the whole history fits inside the keep
        # window).  Pruning preserves the tool-call pairing, so shrink the fat
        # results instead of sending a request the provider rejects for length.
        if estimate_tokens(working, chars_per_token) > max_tokens:
            reduced, extra_pruned = prune_tool_results(working, keep_recent=2, limit_chars=400)
            if extra_pruned:
                working = reduced
                stats["pruned_results"] += extra_pruned
                pruned += extra_pruned
        stats["after_tokens"] = estimate_tokens(working, chars_per_token)
        return working, pruned > 0, stats

    older = conversation[:cut]
    recent = conversation[cut:]
    if not older or not recent:
        stats["after_tokens"] = estimate_tokens(working, chars_per_token)
        return working, pruned > 0, stats

    # Leave room for recent turns; never let the summary eat the whole budget.
    summary_budget_chars = max(800, min(6000, (target // 3) * chars_per_token))
    summary, used_llm = _summarize(client, older, summary_budget_chars, use_llm)
    stats["summarized_messages"] = len(older)
    stats["used_llm"] = used_llm
    summary_message = {
        "role": "user",
        "content": PROMPT_PREFIX + summary,
    }
    compacted = systems + [summary_message] + recent

    # If the summary itself was too fat, trim its body before dropping turns.
    while (
        len(compacted) > 1
        and estimate_tokens(compacted, chars_per_token) > max_tokens
        and len(summary_message["content"]) > 400
    ):
        body = summary_message["content"]
        summary_message["content"] = body[: max(300, int(len(body) * 0.7))] + "\n[summary truncated]"
        compacted = systems + [summary_message] + recent

    # The recent window can exceed the budget by itself (one assistant turn
    # with a wide tool batch).  Shrink its fat results rather than dropping the
    # whole step: the batch and its pairing survive and the model still sees
    # how large each output was.
    if estimate_tokens(compacted, chars_per_token) > max_tokens:
        reduced, extra_pruned = prune_tool_results(recent, keep_recent=2, limit_chars=400)
        if extra_pruned:
            recent = reduced
            stats["pruned_results"] += extra_pruned
            compacted = systems + [summary_message] + recent

    while (
        len(recent) > 2
        and estimate_tokens(compacted, chars_per_token) > max_tokens
    ):
        # Drop one safe chunk, never a lone tool result and never the whole
        # window when only one user turn remains.
        _drop_oldest_step(recent)
        compacted = systems + [summary_message] + recent
        if not recent:
            break

    stats["after_tokens"] = estimate_tokens(compacted, chars_per_token)
    return compacted, True, stats
