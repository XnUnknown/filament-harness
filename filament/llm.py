"""Minimal LLM clients for local, OpenAI-compatible, and Anthropic providers."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace
from typing import Any, Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


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


class MockLLM:
    """Deterministic client for tests/offline runs."""

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


class OllamaClient:
    def __init__(self, model: str, base_url: str, timeout: int = 120, api_key: str | None = None, reasoning_effort: str | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers=headers,
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


class OpenAICompatibleClient:
    """OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: int = 120, reasoning_effort: str | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
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
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)


class AnthropicClient:
    """Anthropic Messages API endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: int = 120, reasoning_effort: str | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode(),
            headers=headers,
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
        content = data.get("content") or []
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(texts)
        return str(content)


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
    timeout = coerce_timeout(getattr(llm_config, "timeout", None), 120, "llm timeout")
    if timeout < 1:
        raise ValueError("llm timeout must be at least 1 second")
    reasoning_effort = getattr(llm_config, "reasoning_effort", None)
    use_ollama_env = bool(getattr(llm_config, "use_ollama_env", False))

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
        return OllamaClient(model=model, base_url=url, timeout=timeout, api_key=api_key, reasoning_effort=reasoning_effort)
    if kind == "openai":
        url = base_url or "https://api.openai.com/v1"
        if not api_key and not _is_local_url(url):
            raise ValueError("api_key is required for openai provider")
        return OpenAICompatibleClient(model=model, base_url=url, api_key=api_key, timeout=timeout, reasoning_effort=reasoning_effort)
    if kind == "anthropic":
        url = base_url or "https://api.anthropic.com"
        if not api_key and not _is_local_url(url):
            raise ValueError("api_key is required for anthropic provider")
        return AnthropicClient(model=model, base_url=url, api_key=api_key, timeout=timeout, reasoning_effort=reasoning_effort)
    raise ValueError(f"unknown llm kind: {kind}")


def make_client(
    kind: str = "auto",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
    reasoning_effort: str | None = None,
) -> LLMClient:
    cfg = SimpleNamespace(
        kind=kind,
        model=model or "llama3.2",
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        use_ollama_env=(os.environ.get("FILAMENT_USE_OLLAMA") == "1"),
    )
    return build_client(cfg)
