"""Focused coding agent wrapper."""
from __future__ import annotations


class CodingAgent:
    def __init__(self, task: str):
        self.task = task

    def run(self, tools) -> str:
        return tools.spawn_agent(self.task)
