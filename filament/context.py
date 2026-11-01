"""Context budget helper: keeps accumulated text under a token cap."""
from __future__ import annotations


class ContextBudget:
    def __init__(self, max_tokens: int = 120_000, chars_per_token: int = 4):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be positive")
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self._chunks: list[str] = []

    def estimate(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)

    def total_tokens(self) -> int:
        return sum(self.estimate(chunk) for chunk in self._chunks)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens())

    def would_fit(self, text: str) -> bool:
        return self.total_tokens() + self.estimate(text) <= self.max_tokens

    def add(self, text: str) -> bool:
        estimate = self.estimate(text)
        if estimate > self.max_tokens:
            text = f"<context omitted: {estimate} tokens exceeds budget>"
            estimate = self.estimate(text)
        while self._chunks and self.total_tokens() + estimate > self.max_tokens:
            self._chunks.pop(0)
        if self.total_tokens() + estimate > self.max_tokens:
            return False
        self._chunks.append(text)
        return True

    def dump(self) -> str:
        return "\n".join(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
