"""Conversational main orchestrator: plans exact agent tasks from natural requests."""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .agent_tools import Toolbox
from .config import HarnessConfig
from .context_compress import compress_messages
from .llm import build_client
from .orchestrator import Orchestrator
from .session import Session, create_session, load_session, save_session, list_sessions as list_session_files


PLAN_SYSTEM = (
    "You are Filament's main orchestrator. Convert a user's coding request into "
    "a JSON plan for focused coding agents. Return ONLY valid JSON. "
    "Schema: {\"summary\": \"short summary\", \"tasks\": ["
    "{\"id\": \"task1\", \"task\": \"exact coding instruction with signature and behavior\", "
    "\"target\": \"relative/output.py\"}]}. "
    "Rules: maximum 6 tasks; keep tasks independent; each task must be exact; "
    "use relative paths only; no markdown."
)


MAX_PLAN_TASKS = 6


def _sanitize_target(value: Any) -> str:
    """Accept only safe relative paths from untrusted planner output."""
    target = str(value or "").strip().replace("\\", "/")
    if "\x00" in target or any(ord(char) < 32 or ord(char) == 127 for char in target):
        return "output.py"
    if not target:
        return "output.py"
    candidate = PurePosixPath(target)
    if (
        target.startswith("/")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in {"", ".", ".."} or ":" in part for part in candidate.parts)
    ):
        return "output.py"
    normalized = str(candidate)
    return normalized if normalized and normalized != "." else "output.py"


def _extract_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("planner response must be a string")
    cleaned = text.strip()
    # Remove a surrounding markdown code fence if the model added one.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in planner response")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        if not isinstance(obj, dict):
            raise ValueError("planner response must be a JSON object")
        return obj
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if not isinstance(parsed, dict):
                raise ValueError("planner response must be a JSON object")
            return parsed
        raise


_SMALL_TALK = {
    "hello", "hi", "hey", "thanks", "thank you", "who are you", "how are you",
    "good morning", "good evening", "what can you do", "help",
}


def is_small_talk(request: str) -> bool:
    cleaned = request.strip().lower().rstrip("!.? ")
    if cleaned in _SMALL_TALK:
        return True
    code_words = {"build", "create", "write", "fix", "test", "run", "read", "analyze", "refactor", "debug", "implement", "verify", "check", "review", "inspect", "validate"}
    words = set(cleaned.split())
    if len(words) <= 4 and not (words & code_words):
        return True
    return False


class ChatOrchestrator:
    def __init__(
        self,
        config: HarnessConfig,
        shell_cwd: str = ".",
        mode: str = "auto",
        session: Session | None = None,
    ):
        self.config = config
        self.shell_cwd = str(Path(shell_cwd).resolve())
        self.mode = mode
        self.client = build_client(config.llm)
        self.toolbox = Toolbox(self.shell_cwd, config)
        self.session = session or create_session(self.shell_cwd, mode)
        self.session.mode = mode
        self.messages: list[dict[str, Any]] = list(self.session.messages)
        self.history: list[str] = []
        for message in self.messages:
            if message.get("role") == "user":
                self.history.append(str(message.get("content", "")))

    def apply_config(self, config: HarnessConfig) -> None:
        """Adopt a validated config and rebuild components that captured it."""
        self.config = config
        self.client = build_client(config.llm)
        self.toolbox = Toolbox(self.shell_cwd, config)

    def plan(self, request: str) -> tuple[str, list[dict[str, Any]]]:
        self.messages.append({"role": "user", "content": request})
        self.history.append(request)
        context = self._context_text(limit_tokens=max(512, self.config.effective_max_tokens // 4))
        prompt = (
            (f"Recent context:\n{context}\n\n" if context else "")
            + f"User request: {request}\n\nReturn the JSON plan now."
        )
        raw = self.client.generate(prompt, system=PLAN_SYSTEM)
        try:
            data = _extract_json(raw)
            summary = str(data.get("summary") or "").strip()
            tasks = data.get("tasks") or []
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("plan has no tasks")
            task_cap = min(self.config.max_tasks, MAX_PLAN_TASKS)
            normalized = []
            for index, item in enumerate(tasks[:task_cap], 1):
                if isinstance(item, str):
                    normalized.append({"id": f"task{index}", "task": item.strip(), "target": "output.py"})
                elif isinstance(item, dict):
                    task_text = str(item.get("task") or item.get("instruction") or "").strip()
                    if not task_text:
                        continue
                    normalized.append(
                        {
                            "id": str(item.get("id") or f"task{index}").strip() or f"task{index}",
                            "task": task_text,
                            "target": _sanitize_target(item.get("target")),
                        }
                    )
            if not normalized:
                raise ValueError("plan tasks were empty")
            self.messages.append({"role": "assistant", "content": f"PLAN: {summary}"})
            self.session.messages = self.messages
            save_session(self.session, self.shell_cwd)
            self.compress_if_needed()
            return summary, normalized
        except (json.JSONDecodeError, ValueError, TypeError):
            self.messages.append({"role": "assistant", "content": "Fallback single-task plan"})
            self.session.messages = self.messages
            save_session(self.session, self.shell_cwd)
            return "Fallback single-task plan", [
                {"id": "task1", "task": request, "target": "output.py"}
            ]

    def run_plan(
        self,
        tasks: list[dict[str, Any]],
        review: bool = True,
        fix: bool = True,
        max_fixes: int | None = None,
    ) -> dict:
        orchestrator = Orchestrator(
            config=self.config,
            shell_cwd=self.shell_cwd,
            allow_outside_target=self.config.allow_outside_target,
        )
        result = orchestrator.run(tasks, "output.py", review=review, fix=fix, max_fixes=max_fixes)
        self.session.files_changed = sorted(set(self.session.files_changed + list(result.get("targets", {}).keys())))
        self.session.touch()
        save_session(self.session, self.shell_cwd)
        self.messages.append({"role": "assistant", "content": f"EXECUTION: {result}"})
        self.compress_if_needed()
        result["plan_summary"] = ""
        return result

    def _context_text(self, limit_tokens: int = 1000) -> str:
        lines = []
        for message in self.messages[-12:]:
            role = message.get("role", "user")
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        max_chars = max(256, limit_tokens * self.config.chars_per_token)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def reflect(self, execution_result: dict) -> tuple[str, list[dict[str, Any]]] | None:
        """Ask the orchestrator whether another pass is needed after execution."""
        from .agent_tools import Toolbox

        _ = Toolbox
        target = execution_result.get("target_file", "output.py")
        snippets = []
        try:
            snippets.append(self.toolbox.read_file(target, max_chars=8000))
        except Exception:  # noqa: BLE001
            snippets.append("(target file unavailable)")
        context = self._context_text()
        prompt = (
            f"Recent context:\n{context}\n\n"
            f"Execution result: {execution_result}\n"
            f"Generated file contents:\n{snippets[0]}\n\n"
            "Decide whether the work is complete. Return JSON only: "
            "{\"action\":\"done\",\"content\":\"short summary\"} or "
            "{\"action\":\"plan\",\"summary\":\"...\",\"tasks\":[...]}."
        )
        system = "You are an autonomous coding orchestrator. Prefer done unless there is a clear defect."
        try:
            raw = self.client.generate(prompt, system=system)
            data = _extract_json(raw)
        except Exception:  # noqa: BLE001
            return None
        action = str(data.get("action") or "").lower()
        if action == "plan":
            tasks = data.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                return None
            normalized = []
            for index, item in enumerate(tasks[:MAX_PLAN_TASKS], 1):
                if isinstance(item, str):
                    normalized.append({"id": f"task{index}", "task": item, "target": target})
                elif isinstance(item, dict):
                    task_text = str(item.get("task") or item.get("instruction") or "").strip()
                    if task_text:
                        normalized.append(
                            {
                                "id": str(item.get("id") or f"task{index}"),
                                "task": task_text,
                                "target": _sanitize_target(item.get("target") or target),
                            }
                        )
            if not normalized:
                return None
            return str(data.get("summary") or "Follow-up plan"), normalized
        return None

    def direct_chat(self, request: str) -> str:
        self.messages.append({"role": "user", "content": request})
        try:
            response = self.client.generate(request, system="You are a helpful coding assistant. Be concise.")
        except Exception as exc:  # noqa: BLE001
            response = f"Unable to respond: {exc}"
        self.messages.append({"role": "assistant", "content": response})
        self.session.messages = self.messages
        save_session(self.session, self.shell_cwd)
        self.compress_if_needed()
        return response

    def is_small_talk(self, request: str) -> bool:
        return is_small_talk(request)

    def compress_if_needed(self) -> bool:
        compressed, changed = compress_messages(
            self.client,
            self.messages,
            self.config.effective_max_tokens,
            self.config.chars_per_token,
        )
        if changed:
            self.messages = compressed
            self.session.messages = self.messages
            save_session(self.session, self.shell_cwd)
        return changed

    def save(self) -> Path:
        self.session.messages = self.messages
        self.session.mode = self.mode
        return save_session(self.session, self.shell_cwd)

    def load_session(self, session_id: str) -> None:
        from .session import load_session as _load

        session = _load(session_id, self.shell_cwd)
        self.session = session
        self.mode = session.mode
        self.messages = list(session.messages)
        self.history = [m["content"] for m in self.messages if m.get("role") == "user"]

    @staticmethod
    def list_sessions(cwd: str | Path) -> list:
        return list_session_files(cwd)

    def run_request(
        self,
        request: str,
        review: bool = True,
        fix: bool = True,
        max_fixes: int | None = None,
    ) -> dict:
        self.history.append(request)
        summary, tasks = self.plan(request)
        result = self.run_plan(tasks, review=review, fix=fix, max_fixes=max_fixes)
        result["plan_summary"] = summary
        return result
