"""Context budget helper: keeps accumulated text under a token cap."""
from __future__ import annotations

CHARS_PER_TOKEN = 4


class ContextBudget:
    def __init__(self, max_tokens: int = 120_000):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * CHARS_PER_TOKEN
        self._chunks: list[str] = []

    def estimate(self, text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    def add(self, text: str) -> None:
        self._chunks.append(text)
        if self.total_tokens() > self.max_tokens:
            # Drop oldest chunks until within budget; keep last chunk at least
            while len(self._chunks) > 1 and self.total_tokens() > self.max_tokens:
                self._chunks.pop(0)

    def total_tokens(self) -> int:
        return sum(self.estimate(c) for c in self._chunks)

    def dump(self) -> str:
        return "\n".join(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
