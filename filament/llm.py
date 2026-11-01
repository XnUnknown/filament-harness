"""Minimal LLM clients for local (Ollama) or mock usage."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol


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
        return f"```python\ndef {name}({params}):\n{body}\n```"


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("FILAMENT_MODEL", "llama3.2")
        self.base_url = (base_url or os.environ.get("FILAMENT_OLLAMA_URL", "http://localhost:11434")).rstrip("/")

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
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data.get("response", "")


def make_client(kind: str = "auto") -> LLMClient:
    if kind == "mock":
        return MockLLM()
    if kind == "ollama":
        return OllamaClient()
    # auto: try ollama env presence, fall back to mock
    if os.environ.get("FILAMENT_USE_OLLAMA") == "1":
        return OllamaClient()
    return MockLLM()
