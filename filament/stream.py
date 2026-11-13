"""Streaming transport helpers for provider chat APIs.

The harness speaks three streaming dialects: OpenAI-style SSE, Anthropic SSE,
and Ollama's newline-delimited JSON.  Everything here is defensive: one
malformed line must not kill an otherwise healthy run.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Iterator


class StreamParseError(RuntimeError):
    """Raised when a provider stream cannot be understood at all."""


def new_call_id() -> str:
    """Return a tool-call id that cannot collide with another generated id.

    Several providers omit ids (Ollama never sends them), so a counter starting
    at 1 restarts on every turn and every turn's first call becomes ``call_1``.
    The id is only used to pair a result with its call, so it must be unique
    across the whole conversation, not just inside one response.
    """
    return "call_" + uuid.uuid4().hex[:12]


def iter_sse_data(response: Any) -> Iterator[str]:
    """Yield ``data:`` payloads from a text/event-stream response.

    ``response`` only needs to be iterable by line, which both
    ``http.client.HTTPResponse`` and test doubles satisfy.
    """
    for raw in response:
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", "replace")
        else:
            line = str(raw)
        line = line.rstrip("\r\n")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            # event:, id:, retry: are not payloads.
            continue
        data = line[5:]
        if data.startswith(" "):
            data = data[1:]
        yield data


def iter_json_lines(response: Any) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a newline-delimited JSON response."""
    for raw in response:
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", "replace")
        else:
            line = str(raw)
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            yield item


def parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str]:
    """Return ``(arguments, error)`` from a provider tool-call payload.

    Providers disagree here: OpenAI sends a JSON string, Ollama and Anthropic
    send an object.  A bad payload becomes a tool error the model can recover
    from instead of an exception that aborts the run.
    """
    if raw is None or raw == "":
        return {}, ""
    if isinstance(raw, dict):
        return raw, ""
    if isinstance(raw, (list, tuple)):
        return {}, "tool arguments must be a JSON object, got a list"
    if not isinstance(raw, str):
        return {}, f"tool arguments must be JSON, got {type(raw).__name__}"
    text = raw.strip()
    if not text:
        return {}, ""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON tool arguments: {exc}"
    if not isinstance(value, dict):
        return {}, f"tool arguments must be a JSON object, got {type(value).__name__}"
    return value, ""


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (ValueError, TypeError):
        return False


def _merge_argument_text(current: str, fragment: str) -> str:
    """Append one argument fragment, tolerating servers that resend the call."""
    if not fragment:
        return current
    if not current or current == fragment:
        return fragment
    # A valid JSON object cannot be followed by more argument text.  When both
    # sides look like complete objects the server resent the whole call instead
    # of a delta; appending would produce "{...}{...}" and fail to parse.
    # The cheap range checks keep normal fragmented streams from re-parsing the
    # accumulated text on every chunk.
    left = fragment.lstrip()
    if (
        left.startswith("{")
        and left.rstrip().endswith("}")
        and current.rstrip().endswith("}")
        and _is_json_object(current.strip())
        and _is_json_object(left.rstrip())
    ):
        return fragment
    return current + fragment


def merge_tool_call_delta(
    accumulator: dict[int, dict[str, Any]],
    index: int,
    payload: dict[str, Any],
) -> None:
    """Fold one streamed OpenAI/Ollama tool-call fragment into ``accumulator``.

    The first fragment carries the id and name; later fragments only carry
    argument text.  Some local servers repeat the whole call on every chunk,
    so ids and names are only accepted once.
    """
    item = accumulator.setdefault(index, {"id": "", "name": "", "arguments": ""})
    call_id = payload.get("id")
    if isinstance(call_id, str) and call_id and not item["id"]:
        item["id"] = call_id
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name and not item["name"]:
            item["name"] = name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            item["arguments"] = _merge_argument_text(item["arguments"], arguments)
        elif isinstance(arguments, dict):
            item["arguments"] = json.dumps(arguments)
    # Ollama sometimes sends the call flat instead of under "function".
    name = payload.get("name")
    if isinstance(name, str) and name and not item["name"]:
        item["name"] = name
    arguments = payload.get("arguments")
    if isinstance(arguments, dict) and not item["arguments"]:
        item["arguments"] = json.dumps(arguments)
    elif isinstance(arguments, str) and not item["arguments"]:
        item["arguments"] = arguments


def finish_tool_calls(accumulator: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn accumulated fragments into ``{id, name, arguments}`` dicts."""
    calls: list[dict[str, Any]] = []
    for index in sorted(accumulator):
        item = accumulator[index]
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        arguments, error = parse_tool_arguments(item.get("arguments"))
        calls.append(
            {
                "id": str(item.get("id") or "") or new_call_id(),
                "name": name,
                "arguments": arguments,
                "argument_error": error,
            }
        )
    return calls
