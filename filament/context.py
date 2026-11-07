"""Context budget helper: keeps accumulated text under a token cap."""
from __future__ import annotations

import threading

class ContextBudget:
    def __init__(self, max_tokens: int = 120_000, chars_per_token: int = 4):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be positive")
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self._chunks: list[str] = []
        self._lock = threading.RLock()

    def estimate(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)

    def total_tokens(self) -> int:
        with self._lock:
            return sum(self.estimate(chunk) for chunk in self._chunks)

    def remaining_tokens(self) -> int:
        with self._lock:
            return max(0, self.max_tokens - sum(self.estimate(chunk) for chunk in self._chunks))

    def would_fit(self, text: str) -> bool:
        with self._lock:
            return sum(self.estimate(chunk) for chunk in self._chunks) + self.estimate(text) <= self.max_tokens

    def add(self, text: str) -> bool:
        with self._lock:
            estimate = self.estimate(text)
            if estimate > self.max_tokens:
                text = f"<context omitted: {estimate} tokens exceeds budget>"
                estimate = self.estimate(text)
            if estimate > self.max_tokens:
                return False
            while self._chunks and sum(self.estimate(chunk) for chunk in self._chunks) + estimate > self.max_tokens:
                self._chunks.pop(0)
            if sum(self.estimate(chunk) for chunk in self._chunks) + estimate > self.max_tokens:
                return False
            self._chunks.append(text)
            return True

    def dump(self) -> str:
        with self._lock:
            return "\n".join(self._chunks)

    def reset(self) -> None:
        with self._lock:
            self._chunks.clear()
