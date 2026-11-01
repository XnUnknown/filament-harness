"""Minimal LLM clients for local, OpenAI-compatible, and Anthropic providers."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


class MockLLM:
    """Deterministic client for tests/offline runs."""

    def generate(self, prompt: str, system: str | None = None) -> str:
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
    def __init__(self, model: str, base_url: str, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc
        return data.get("response", "")


class OpenAICompatibleClient:
    """OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False}
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
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"openai-compatible request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"openai-compatible request failed: {exc}") from exc

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

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

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
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"anthropic request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"anthropic request failed: {exc}") from exc

        content = data.get("content") or []
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(texts)
        return str(content)


def _is_local_url(url: str | None) -> bool:
    return bool(url and ("localhost" in url or "127.0.0.1" in url))


def build_client(llm_config: Any) -> LLMClient:
    kind = getattr(llm_config, "kind", None) or "auto"
    kind = kind.lower().strip()
    model = getattr(llm_config, "model", None) or "llama3.2"
    base_url = getattr(llm_config, "base_url", None)
    api_key = getattr(llm_config, "api_key", None) or os.environ.get("FILAMENT_API_KEY")
    timeout = getattr(llm_config, "timeout", 120) or 120
    use_ollama_env = bool(getattr(llm_config, "use_ollama_env", False))

    if kind == "auto":
        if use_ollama_env or os.environ.get("FILAMENT_USE_OLLAMA") == "1":
            kind = "ollama"
        else:
            kind = "mock"

    if kind == "mock":
        return MockLLM()
    if kind == "ollama":
        url = base_url or os.environ.get("FILAMENT_OLLAMA_URL") or "http://localhost:11434"
        return OllamaClient(model=model, base_url=url, timeout=timeout)
    if kind == "openai":
        url = base_url or "https://api.openai.com/v1"
        if not api_key and not _is_local_url(url):
            raise ValueError("api_key is required for openai provider")
        return OpenAICompatibleClient(model=model, base_url=url, api_key=api_key, timeout=timeout)
    if kind == "anthropic":
        url = base_url or "https://api.anthropic.com"
        if not api_key and not _is_local_url(url):
            raise ValueError("api_key is required for anthropic provider")
        return AnthropicClient(model=model, base_url=url, api_key=api_key, timeout=timeout)
    raise ValueError(f"unknown llm kind: {kind}")


def make_client(
    kind: str = "auto",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
) -> LLMClient:
    cfg = SimpleNamespace(
        kind=kind,
        model=model or "llama3.2",
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        use_ollama_env=(os.environ.get("FILAMENT_USE_OLLAMA") == "1"),
    )
    return build_client(cfg)
