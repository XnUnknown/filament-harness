"""LLM clients: text completion plus tool-calling chat for every provider.

Two surfaces are exposed:

``generate(prompt, system)``
    The original one-shot completion used by the batch runner, review phase and
    context summarizer. Unchanged on purpose.

``chat(messages, tools, on_delta)``
    The agent-loop surface. Messages use one neutral format and every provider
    adapter converts to its own wire format. When ``on_delta`` is supplied the
    request streams and each text fragment is forwarded as it arrives; tool
    calls are accumulated across chunks.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from .stream import (
    finish_tool_calls,
    iter_json_lines,
    iter_sse_data,
    merge_tool_call_delta,
    parse_tool_arguments,
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    argument_error: str = ""

    def to_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "argument_error": self.argument_error,
        }


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""


class LLMClient(Protocol):
    supports_tools: bool
    stream: bool

    def generate(self, prompt: str, system: str | None = None) -> str: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ChatResult: ...


# A completion is small; anything near these caps means a misbehaving or hostile
# endpoint. Bounding the read keeps one bad request from exhausting the harness.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_ERROR_BODY_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 64 * 1024


def _read_body(resp: Any, provider: str, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    """Read at most ``limit`` bytes from a provider response."""
    headers = getattr(resp, "headers", None)
    declared = headers.get("Content-Length") if headers is not None else None
    if declared is not None:
        try:
            declared_size = int(str(declared).strip())
        except (TypeError, ValueError):
            declared_size = None
        if declared_size is not None and declared_size > limit:
            raise RuntimeError(
                f"{provider} response exceeded {limit} bytes (Content-Length: {declared_size})"
            )
    body = bytearray()
    while len(body) <= limit:
        chunk = resp.read(min(_READ_CHUNK_BYTES, limit + 1 - len(body)))
        if not chunk:
            break
        body.extend(chunk)
    if len(body) > limit:
        raise RuntimeError(f"{provider} response exceeded {limit} bytes")
    return bytes(body)


def _read_error_body(exc: urllib.error.HTTPError, limit: int = MAX_ERROR_BODY_BYTES) -> str:
    """Read a bounded prefix of an HTTP error body for diagnostics."""
    try:
        raw = exc.read(limit + 1)
    except (OSError, ValueError):
        return ""
    text = raw[:limit].decode(errors="replace")
    if len(raw) > limit:
        text += " [truncated]"
    return text


def _text(content: Any) -> str:
    """Flatten provider content that may be a string or a list of blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "".join(parts)
    return str(content)


def _cancelled(cancel: Callable[[], bool] | None) -> bool:
    try:
        return bool(cancel and cancel())
    except Exception:  # noqa: BLE001 - a broken cancel hook must not break a run
        return False


def _content_type(resp: Any) -> str:
    """Return a lower-cased Content-Type header, or an empty string."""
    headers = getattr(resp, "headers", None)
    if headers is None:
        return ""
    try:
        value = headers.get("Content-Type", "")
    except AttributeError:
        return ""
    return str(value or "").lower()


def _stream_error(provider: str, value: Any) -> RuntimeError:
    """Build an exception from an in-band provider error object."""
    if isinstance(value, dict):
        text = value.get("message") or value.get("error") or value.get("detail")
        if not isinstance(text, str) or not text.strip():
            text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    text = " ".join(str(text).split())[:2000]
    return RuntimeError(f"{provider} stream error: {text or 'unknown error'}")


def _tool_schemas(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert neutral tool specs to the OpenAI function-tool envelope."""
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return result


class MockLLM:
    """Deterministic client for tests and offline runs.

    ``script`` items may be :class:`ChatResult` objects or dicts shaped like
    ``{"content": "...", "tool_calls": [{"name": ..., "arguments": {...}}]}``.
    Once the script is exhausted a plain reply is returned, which terminates an
    agent loop cleanly.
    """

    supports_tools = True
    stream = False

    def __init__(self, script: list[Any] | None = None, default_reply: str | None = None):
        self._script = list(script or [])
        self.default_reply = default_reply
        self._lock = threading.Lock()
        self.requests: list[list[dict[str, Any]]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        if system and "helpful coding assistant" in system.lower():
            return "Hello! I am Filament. Ask me to plan, generate, or fix code."
        import re

        m = re.search(r"named\s+(\w+)\s*\(([^)]*)\)", prompt)
        name = m.group(1) if m else "generated"
        params = m.group(2).strip() if m else ""
        body = "    return None"
        low = name.lower()
        if low == "add":
            body = "    return a + b"
        elif low == "mul":
            body = "    return a * b"
        elif low == "sub":
            body = "    return a - b"
        return f"```python\ndef {name}({params}):\n{body}\n```"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ChatResult:
        with self._lock:
            self.requests.append([dict(message) for message in messages])
            item = self._script.pop(0) if self._script else None
        if item is None:
            last = messages[-1] if messages else {}
            if last.get("role") == "tool":
                content = "Tool result received. Task complete."
            else:
                content = self.default_reply or "MockLLM: no scripted response."
            result = ChatResult(content=content, finish_reason="stop")
        elif isinstance(item, ChatResult):
            result = item
        elif isinstance(item, dict):
            calls: list[ToolCall] = []
            for index, call in enumerate(item.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                calls.append(
                    ToolCall(
                        id=str(call.get("id") or f"call_{index + 1}"),
                        name=str(call.get("name") or ""),
                        arguments=dict(call.get("arguments") or {}),
                        argument_error=str(call.get("argument_error") or ""),
                    )
                )
            result = ChatResult(
                content=str(item.get("content") or ""),
                tool_calls=calls,
                finish_reason=str(item.get("finish_reason") or "stop"),
            )
        else:
            result = ChatResult(content=str(item), finish_reason="stop")
        if on_delta and result.content:
            on_delta(result.content)
        return result


class OllamaClient:
    supports_tools = True

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: int = 120,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        stream: bool = True,
        max_output_tokens: int = 4096,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.max_output_tokens = max_output_tokens

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -- one-shot completion (legacy surface) --------------------------------
    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(_read_body(resp, "ollama").decode())
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            raise RuntimeError(f"ollama request failed: HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("ollama response was not a JSON object")
        return str(data.get("response", ""))

    # -- tool-calling chat ----------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_ollama_message(message) for message in messages],
            "stream": bool(on_delta) and self.stream,
            "options": {"num_predict": int(self.max_output_tokens)},
        }
        schemas = _tool_schemas(tools)
        if schemas:
            payload["tools"] = schemas
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if payload["stream"]:
                    return self._read_stream(resp, on_delta, cancel)
                data = json.loads(_read_body(resp, "ollama").decode())
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            raise RuntimeError(f"ollama request failed: HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("ollama response was not a JSON object")
        return self._result_from_payload(data)

    def _result_from_payload(self, data: dict[str, Any]) -> ChatResult:
        message = data.get("message") or {}
        content = _text(message.get("content"))
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            if not isinstance(item, dict):
                continue
            function = item.get("function") if isinstance(item.get("function"), dict) else item
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            arguments, error = parse_tool_arguments(function.get("arguments"))
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{index + 1}"),
                    name=name,
                    arguments=arguments,
                    argument_error=error,
                )
            )
        return ChatResult(
            content=content,
            tool_calls=calls,
            finish_reason=str(data.get("done_reason") or ("tool_calls" if calls else "stop")),
            usage={
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            },
            model=str(data.get("model") or self.model),
        )

    def _read_stream(
        self,
        resp: Any,
        on_delta: Callable[[str], None] | None,
        cancel: Callable[[], bool] | None,
    ) -> ChatResult:
        parts: list[str] = []
        calls: list[ToolCall] = []
        usage: dict[str, Any] = {}
        finish = ""
        for chunk in iter_json_lines(resp):
            if _cancelled(cancel):
                break
            error = chunk.get("error")
            if error:
                raise _stream_error("ollama", error)
            message = chunk.get("message")
            if not isinstance(message, dict):
                message = {}
            text = _text(message.get("content"))
            if text:
                parts.append(text)
                if on_delta:
                    on_delta(text)
            for item in message.get("tool_calls") or []:
                if not isinstance(item, dict):
                    continue
                function = item.get("function") if isinstance(item.get("function"), dict) else item
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                arguments, error = parse_tool_arguments(function.get("arguments"))
                calls.append(
                    ToolCall(
                        id=str(item.get("id") or f"call_{len(calls) + 1}"),
                        name=name,
                        arguments=arguments,
                        argument_error=error,
                    )
                )
            if chunk.get("done"):
                finish = str(chunk.get("done_reason") or "stop")
                usage = {
                    "prompt_tokens": chunk.get("prompt_eval_count"),
                    "completion_tokens": chunk.get("eval_count"),
                }
                break
        return ChatResult(
            content="".join(parts),
            tool_calls=calls,
            finish_reason=finish or ("tool_calls" if calls else "stop"),
            usage=usage,
            model=self.model,
        )


def _ollama_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role") or "user"
    out: dict[str, Any] = {"role": role, "content": _text(message.get("content"))}
    if role == "assistant" and message.get("tool_calls"):
        out["tool_calls"] = [
            {
                "function": {
                    "name": call.get("name"),
                    "arguments": call.get("arguments") or {},
                }
            }
            for call in message["tool_calls"]
            if isinstance(call, dict) and call.get("name")
        ]
    return out


class OpenAICompatibleClient:
    """OpenAI-compatible chat completions endpoint."""

    supports_tools = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 120,
        reasoning_effort: str | None = None,
        stream: bool = True,
        max_output_tokens: int = 4096,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.max_output_tokens = max_output_tokens

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(_read_body(resp, "openai-compatible").decode())
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            raise RuntimeError(f"openai-compatible request failed: HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"openai-compatible request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("openai-compatible response was not a JSON object")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("openai-compatible response contained no choices")
        message = choices[0].get("message", {})
        return _text(message.get("content"))

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ChatResult:
        stream = bool(on_delta) and self.stream
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_openai_message(message) for message in messages],
            "stream": stream,
        }
        if self.max_output_tokens:
            payload["max_tokens"] = int(self.max_output_tokens)
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        schemas = _tool_schemas(tools)
        if schemas:
            payload["tools"] = schemas
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                # Servers sometimes ignore stream=true; honour the body type.
                if stream and "event-stream" in _content_type(resp):
                    return self._read_stream(resp, on_delta, cancel)
                data = json.loads(_read_body(resp, "openai-compatible").decode())
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            raise RuntimeError(f"openai-compatible request failed: HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"openai-compatible request failed: {exc}") from exc
        return self._result_from_payload(data)

    def _result_from_payload(self, data: dict[str, Any]) -> ChatResult:
        if not isinstance(data, dict):
            raise RuntimeError("openai-compatible response was not a JSON object")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("openai-compatible response contained no choices")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            if not isinstance(item, dict):
                continue
            function = item.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            arguments, error = parse_tool_arguments(function.get("arguments"))
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{index + 1}"),
                    name=name,
                    arguments=arguments,
                    argument_error=error,
                )
            )
        usage = data.get("usage") or {}
        return ChatResult(
            content=_text(message.get("content")),
            tool_calls=calls,
            finish_reason=str(choice.get("finish_reason") or ("tool_calls" if calls else "stop")),
            usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
            model=str(data.get("model") or self.model),
        )

    def _read_stream(
        self,
        resp: Any,
        on_delta: Callable[[str], None] | None,
        cancel: Callable[[], bool] | None,
    ) -> ChatResult:
        parts: list[str] = []
        accumulator: dict[int, dict[str, Any]] = {}
        finish = ""
        usage: dict[str, Any] = {}
        for data in iter_sse_data(resp):
            if _cancelled(cancel):
                break
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            # A provider can fail after the response has started; the error
            # arrives as an ordinary event, not an HTTP status.  Returning an
            # empty result would look like a model that said nothing.
            error = chunk.get("error")
            if error:
                raise _stream_error("openai-compatible", error)
            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict) and chunk_usage:
                usage = {
                    "prompt_tokens": chunk_usage.get("prompt_tokens"),
                    "completion_tokens": chunk_usage.get("completion_tokens"),
                }
            for choice in chunk.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                text = _text(delta.get("content"))
                if text:
                    parts.append(text)
                    if on_delta:
                        on_delta(text)
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    try:
                        index = int(call.get("index", 0))
                    except (TypeError, ValueError):
                        index = 0
                    merge_tool_call_delta(accumulator, index, call)
                if choice.get("finish_reason"):
                    finish = str(choice["finish_reason"])
        calls = [
            ToolCall(
                id=str(item.get("id") or f"call_{index + 1}"),
                name=str(item.get("name") or ""),
                arguments=dict(item.get("arguments") or {}),
                argument_error=str(item.get("argument_error") or ""),
            )
            for index, item in enumerate(finish_tool_calls(accumulator))
        ]
        return ChatResult(
            content="".join(parts),
            tool_calls=calls,
            finish_reason=finish or ("tool_calls" if calls else "stop"),
            usage=usage,
            model=self.model,
        )


def _openai_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role") or "user"
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": str(message.get("tool_call_id") or ""),
            "content": _text(message.get("content")),
        }
    out: dict[str, Any] = {"role": role, "content": _text(message.get("content"))}
    if role == "assistant" and message.get("tool_calls"):
        out["content"] = out["content"] or None
        out["tool_calls"] = [
            {
                "id": str(call.get("id") or f"call_{index + 1}"),
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for index, call in enumerate(message["tool_calls"])
            if isinstance(call, dict) and call.get("name")
        ]
    return out


def _block_index(event: dict[str, Any]) -> int:
    """Return an Anthropic content-block index, tolerating malformed values."""
    try:
        return int(event.get("index", 0))
    except (TypeError, ValueError):
        return 0


class AnthropicClient:
    """Anthropic Messages API endpoint with tool use."""

    supports_tools = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 120,
        reasoning_effort: str | None = None,
        stream: bool = True,
        max_output_tokens: int = 4096,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.max_output_tokens = max_output_tokens

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(_read_body(resp, "anthropic").decode())
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            raise RuntimeError(f"anthropic request failed: HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"anthropic request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("anthropic response was not a JSON object")
        return _text(data.get("content"))

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ChatResult:
        system, conversation = _anthropic_messages(messages)
        stream = bool(on_delta) and self.stream
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(self.max_output_tokens),
            "messages": conversation,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        schemas = [
            {
                "name": tool["name"],
                "description": str(tool.get("description") or ""),
                "input_schema": tool.get("parameters")
                or {"type": "object", "properties": {}},
            }
            for tool in tools or []
            if tool.get("name")
        ]
        if schemas:
            payload["tools"] = schemas
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if stream and "event-stream" in _content_type(resp):
                    return self._read_stream(resp, on_delta, cancel)
                data = json.loads(_read_body(resp, "anthropic").decode())
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            raise RuntimeError(f"anthropic request failed: HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"anthropic request failed: {exc}") from exc
        return self._result_from_payload(data)

    def _result_from_payload(self, data: dict[str, Any]) -> ChatResult:
        if not isinstance(data, dict):
            raise RuntimeError("anthropic response was not a JSON object")
        content = data.get("content") or []
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                name = str(block.get("name") or "").strip()
                if not name:
                    continue
                arguments = block.get("input")
                if not isinstance(arguments, dict):
                    arguments, error = parse_tool_arguments(arguments)
                else:
                    error = ""
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or f"call_{len(calls) + 1}"),
                        name=name,
                        arguments=arguments,
                        argument_error=error,
                    )
                )
        usage = data.get("usage") or {}
        return ChatResult(
            content="".join(texts),
            tool_calls=calls,
            finish_reason=str(data.get("stop_reason") or ("tool_calls" if calls else "stop")),
            usage={
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            },
            model=str(data.get("model") or self.model),
        )

    def _read_stream(
        self,
        resp: Any,
        on_delta: Callable[[str], None] | None,
        cancel: Callable[[], bool] | None,
    ) -> ChatResult:
        blocks: dict[int, dict[str, Any]] = {}
        finish = ""
        usage: dict[str, Any] = {}
        for data in iter_sse_data(resp):
            if _cancelled(cancel):
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "content_block_start":
                block = event.get("content_block")
                if not isinstance(block, dict):
                    block = {}
                # The official API sends an empty ``input`` object in the start
                # block and streams the real arguments as input_json_delta.
                # Some Anthropic-compatible servers send the whole input here
                # with no deltas; keep it as a fallback so the call is not
                # executed with empty arguments.
                start_input = block.get("input")
                blocks[_block_index(event)] = {
                    "type": block.get("type"),
                    "text": str(block.get("text") or ""),
                    "id": block.get("id") or "",
                    "name": block.get("name") or "",
                    "input": start_input if isinstance(start_input, dict) else None,
                    "json": "",
                }
            elif kind == "content_block_delta":
                delta = event.get("delta")
                if not isinstance(delta, dict):
                    continue
                index = _block_index(event)
                item = blocks.setdefault(
                    index,
                    {
                        "type": "text",
                        "text": "",
                        "id": "",
                        "name": "",
                        "input": None,
                        "json": "",
                    },
                )
                if delta.get("type") == "text_delta":
                    text = str(delta.get("text") or "")
                    item["text"] += text
                    if text and on_delta:
                        on_delta(text)
                elif delta.get("type") == "input_json_delta":
                    item["json"] += str(delta.get("partial_json") or "")
            elif kind == "message_delta":
                delta = event.get("delta")
                if isinstance(delta, dict):
                    finish = str(delta.get("stop_reason") or "")
                event_usage = event.get("usage")
                if isinstance(event_usage, dict) and event_usage:
                    usage = {
                        "prompt_tokens": event_usage.get("input_tokens"),
                        "completion_tokens": event_usage.get("output_tokens"),
                    }
            elif kind == "error":
                error = event.get("error")
                message = (
                    error.get("message")
                    if isinstance(error, dict)
                    else None
                ) or "anthropic stream error"
                raise RuntimeError(f"anthropic request failed: {message}")
        texts = [item["text"] for item in blocks.values() if item.get("type") == "text"]
        calls: list[ToolCall] = []
        for item in blocks.values():
            if item.get("type") != "tool_use" or not item.get("name"):
                continue
            raw_json = str(item.get("json") or "")
            if raw_json.strip():
                arguments, error = parse_tool_arguments(raw_json)
            elif isinstance(item.get("input"), dict):
                arguments, error = dict(item["input"]), ""
            else:
                arguments, error = {}, ""
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{len(calls) + 1}"),
                    name=str(item["name"]),
                    arguments=arguments,
                    argument_error=error,
                )
            )
        return ChatResult(
            content="".join(texts),
            tool_calls=calls,
            finish_reason=finish or ("tool_calls" if calls else "stop"),
            usage=usage,
            model=self.model,
        )


def _anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role") or "user"
        content = _text(message.get("content"))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": str(message.get("tool_call_id") or ""),
                "content": content or "(empty)",
            }
            if message.get("is_error"):
                block["is_error"] = True
            if (
                conversation
                and conversation[-1]["role"] == "user"
                and all(part.get("type") == "tool_result" for part in conversation[-1]["content"])
            ):
                conversation[-1]["content"].append(block)
            else:
                conversation.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict) or not call.get("name"):
                    continue
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id") or f"call_{len(blocks) + 1}"),
                        "name": str(call["name"]),
                        "input": call.get("arguments") or {},
                    }
                )
            conversation.append(
                {"role": "assistant", "content": blocks or [{"type": "text", "text": " "}]}
            )
            continue
        # user
        blocks = [{"type": "text", "text": content or "(empty)"}]
        if conversation and conversation[-1]["role"] == "user":
            # Strict Anthropic-compatible backends reject two user turns in a
            # row (Bedrock returns 400), so merge instead.  This is also the
            # path that puts a follow-up question after a tool_result turn.
            conversation[-1]["content"].extend(blocks)
        else:
            conversation.append({"role": "user", "content": blocks})
    return "\n\n".join(system_parts), conversation


def _is_local_url(url: str | None) -> bool:
    if not url:
        return False
    text = str(url).strip()
    if not text:
        return False
    host = None
    try:
        host = urllib.parse.urlsplit(text).hostname
    except (ValueError, AttributeError):
        host = None
    if host is None:
        host = text.split("/", 1)[0]
        host = host.strip()
        if host.startswith("["):
            closing = host.find("]")
            if closing != -1:
                host = host[1:closing]
            else:
                host = host.strip("[]")
        elif host.count(":") <= 1 and ":" in host:
            host = host.split(":", 1)[0]
        host = host.strip("[]").strip()
    host = str(host).lower().strip("[]")
    return host in {"localhost", "127.0.0.1", "::1"}


def build_client(llm_config: Any) -> LLMClient:
    # config imports this module for _is_local_url; import lazily to avoid a cycle.
    from .config import coerce_timeout

    kind = getattr(llm_config, "kind", None) or "auto"
    kind = kind.lower().strip()
    model = getattr(llm_config, "model", None) or "llama3.2"
    base_url = getattr(llm_config, "base_url", None)
    api_key = getattr(llm_config, "api_key", None)
    timeout = int(coerce_timeout(getattr(llm_config, "timeout", None), 120, "llm timeout"))
    if timeout < 1:
        raise ValueError("llm timeout must be at least 1 second")
    reasoning_effort = getattr(llm_config, "reasoning_effort", None)
    use_ollama_env = bool(getattr(llm_config, "use_ollama_env", False))
    stream = bool(getattr(llm_config, "stream", True))
    max_output_tokens = int(getattr(llm_config, "max_output_tokens", 4096) or 4096)

    if kind == "auto":
        if use_ollama_env or os.environ.get("FILAMENT_USE_OLLAMA") == "1":
            kind = "ollama"
        else:
            kind = "mock"

    if api_key is None:
        if kind == "ollama":
            api_key = os.environ.get("FILAMENT_API_KEY") or os.environ.get("OLLAMA_API_KEY")
        else:
            api_key = os.environ.get("FILAMENT_API_KEY")

    if kind == "mock":
        return MockLLM()
    if kind == "ollama":
        url = base_url or os.environ.get("FILAMENT_OLLAMA_URL") or "http://localhost:11434"
        return OllamaClient(
            model=model,
            base_url=url,
            timeout=timeout,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            stream=stream,
            max_output_tokens=max_output_tokens,
        )
    if kind == "openai":
        url = base_url or "https://api.openai.com/v1"
        if not api_key and not _is_local_url(url):
            raise ValueError("api_key is required for openai provider")
        return OpenAICompatibleClient(
            model=model,
            base_url=url,
            api_key=api_key,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            stream=stream,
            max_output_tokens=max_output_tokens,
        )
    if kind == "anthropic":
        url = base_url or "https://api.anthropic.com"
        if not api_key and not _is_local_url(url):
            raise ValueError("api_key is required for anthropic provider")
        return AnthropicClient(
            model=model,
            base_url=url,
            api_key=api_key,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            stream=stream,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"unknown llm kind: {kind}")


def make_client(
    kind: str = "auto",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
    reasoning_effort: str | None = None,
    stream: bool = True,
    max_output_tokens: int = 4096,
) -> LLMClient:
    cfg = SimpleNamespace(
        kind=kind,
        model=model or "llama3.2",
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        use_ollama_env=(os.environ.get("FILAMENT_USE_OLLAMA") == "1"),
        stream=stream,
        max_output_tokens=max_output_tokens,
    )
    return build_client(cfg)
