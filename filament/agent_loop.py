"""The Filament agent loop.

One loop per agent, modelled on the way mature coding agents work:

    think -> call tools -> read results -> think again -> ... -> finish

Every agent owns a :class:`~filament.session.Session` and its own message list.
A subagent is a full second loop with a smaller budget; when it finishes, its
final answer is handed back to the parent as an ordinary tool result and the
parent decides what to do next.  Nothing here pretends a batch of one-shot
completions is an agent.
"""
from __future__ import annotations

import json
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .compaction import compact_messages, estimate_tokens
from .config import HarnessConfig
from .llm import ChatResult, LLMClient, ToolCall, build_client
from .session import Session, create_session, save_session
from .toolkit import (
    AskUser,
    FinishTask,
    ToolError,
    ToolRegistry,
)


EventSink = Callable[[dict[str, Any]], None]


@dataclass
class LoopResult:
    status: str
    content: str = ""
    steps: int = 0
    tokens: int = 0
    session_id: str = ""
    question: dict[str, Any] | None = None
    error: str | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"done", "awaiting_input"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "content": self.content,
            "steps": self.steps,
            "tokens": self.tokens,
            "session_id": self.session_id,
            "question": self.question,
            "error": self.error,
            "todos": self.todos,
            "files_changed": self.files_changed,
        }


def build_system_prompt(
    config: HarnessConfig,
    workspace: Path,
    *,
    depth: int = 0,
    read_only: bool = False,
    model: str = "",
) -> str:
    kind = getattr(config.llm, "kind", "auto")
    lines = [
        "You are Filament, an autonomous coding agent working directly in the user's workspace.",
        "",
        "Environment",
        f"- Workspace: {workspace}",
        f"- Platform: {platform.system()} {platform.release()}",
        f"- Provider/model: {kind}/{model or config.llm.model}",
        f"- Today: {time.strftime('%Y-%m-%d')}",
        "",
        "How you work",
        "- You run in a loop: reason briefly, call tools, read the results, repeat. Never claim work you have not done.",
        "- Inspect before changing. Read the relevant files, or use grep/glob to locate symbols, before editing.",
        "- Change existing code with edit_file or multi_edit. Use write_file only for new files or deliberate full rewrites.",
        "- Verify with the project's tests, linter, or build when they exist. Fix failures before finishing.",
        "- For multi-step work, keep todo_write current; mark exactly one item in_progress at a time.",
        "- If a requirement is ambiguous, a choice materially changes the design, or an action is destructive, call ask_user and wait. Ask early instead of guessing.",
        "- Delegate independent work to subagents with spawn_agent or spawn_agents. They have their own context and session, and return a summary. Give complete instructions.",
        "- Keep tool arguments small. Do not dump whole files when a targeted read or grep will do.",
        "- Never invent file contents, command output, or test results.",
        "- Finish only when the task is genuinely complete: call finish with what changed and how it was verified.",
        "",
        "If native tool calls are unavailable, reply with exactly one JSON object: "
        '{"tool": "<name>", "arguments": {...}} or {"final": "<answer>"}. No prose around it.',
    ]
    if read_only:
        lines.extend(
            [
                "",
                "Plan mode",
                "You may read, search, inspect, and ask questions, but you may not modify files or run commands.",
                "Produce a concrete plan with file-level steps and call finish with the plan.",
            ]
        )
    if depth > 0:
        lines.extend(
            [
                "",
                "Subagent rules",
                "You are a subagent working on one delegated task.",
                "Do not ask the user questions; make a reasonable assumption, state it in your report, and continue.",
                "When done, call finish with a concise report: what changed, where, and what remains.",
            ]
        )
    return "\n".join(lines)


class AgentLoop:
    """A single agent's state machine and conversation."""

    def __init__(
        self,
        config: HarnessConfig,
        workspace: str | Path,
        session: Session,
        client: LLMClient | None = None,
        event_sink: EventSink | None = None,
        *,
        depth: int = 0,
        read_only: bool | None = None,
        max_steps: int | None = None,
        budget_tokens: int | None = None,
        subagent_id: str = "",
        cancel_event: threading.Event | None = None,
        system_extra: str = "",
    ):
        self.config = config
        self.workspace = Path(workspace).expanduser().resolve()
        self.session = session
        self.client = client or build_client(config.llm)
        self.sink = event_sink
        self.depth = max(0, depth)
        self.read_only = bool(read_only) if read_only is not None else session.mode == "plan"
        self.max_steps = max(1, int(max_steps or (config.subagent_max_steps if depth > 0 else config.max_steps)))
        self.budget_tokens = max(
            1024,
            int(budget_tokens or (config.subagent_max_tokens if depth > 0 else config.effective_max_tokens)),
        )
        self.subagent_id = subagent_id
        self._cancel = cancel_event or threading.Event()
        self._files: list[str] = list(session.files_changed)
        self._lock = threading.RLock()
        usage = session.usage if isinstance(session.usage, dict) else {}
        try:
            self._prior_steps = int(usage.get("steps", 0) or 0)
        except (TypeError, ValueError):
            self._prior_steps = 0
        # The step limit is per turn.  Starting from the session total meant a
        # session that once reached max_steps returned max_steps immediately on
        # every later message, without ever calling the model again.
        self.steps = 0
        try:
            self.compactions = int(usage.get("compactions", 0) or 0)
        except (TypeError, ValueError):
            self.compactions = 0

        self.messages: list[dict[str, Any]] = list(session.messages)
        # Text streamed during the current model call, kept so a failed request
        # can still close the assistant turn with what the user already saw.
        self._stream_parts: list[str] = []
        system_prompt = build_system_prompt(
            config,
            self.workspace,
            depth=self.depth,
            read_only=self.read_only,
            model=str(getattr(self.client, "model", "") or config.llm.model),
        )
        if system_extra:
            system_prompt += "\n\nAdditional instructions\n" + system_extra.strip()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": system_prompt}
        else:
            self.messages.insert(0, {"role": "system", "content": system_prompt})

        self.registry = ToolRegistry(
            self.workspace,
            config,
            depth=self.depth,
            read_only=self.read_only,
            allow_questions=depth == 0,
            allow_subagents=depth < config.max_depth,
            spawn_fn=self._spawn_agent,
            on_file_change=self._on_file_change,
            on_todos=self._on_todos,
            todos=session.todos,
        )
        self.tokens = estimate_tokens(self.messages, config.chars_per_token)

    # -- events -------------------------------------------------------------
    def _emit(self, kind: str, **payload: Any) -> None:
        if not self.sink:
            return
        event: dict[str, Any] = {"type": kind, "session_id": self.session.session_id}
        if self.subagent_id:
            event["subagent_id"] = self.subagent_id
        event.update(payload)
        try:
            self.sink(event)
        except Exception:  # noqa: BLE001 - a broken UI must not kill a run
            pass

    def _emit_context(self) -> None:
        self._emit(
            "context",
            tokens=self.tokens,
            limit=self.budget_tokens,
            messages=len(self.messages),
            steps=self.steps,
            compactions=self.compactions,
        )

    def _status(self, state: str) -> None:
        self._emit("status", state=state)

    # -- lifecycle ----------------------------------------------------------
    def run(self, message: str, system_extra: str = "") -> LoopResult:
        message = str(message or "").strip()
        if not message:
            raise ValueError("message must not be empty")
        if system_extra:
            # Applied for this run only; the durable prompt is rebuilt next time.
            self.messages[0]["content"] += "\n\nAdditional instructions\n" + system_extra.strip()
        self.messages.append({"role": "user", "content": message})
        if not self.session.title:
            first_line = message.splitlines()[0].strip()
            self.session.title = first_line[:80]
        self.session.status = "running"
        self.session.pending_question = None
        self._sync()
        return self._loop()

    def resume(self, answer: str) -> LoopResult:
        pending = self.session.pending_question
        if not pending:
            raise ValueError("no pending question to answer")
        self.messages = list(self.session.messages)
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(
                0,
                {
                    "role": "system",
                    "content": build_system_prompt(
                        self.config,
                        self.workspace,
                        depth=self.depth,
                        read_only=self.read_only,
                        model=str(getattr(self.client, "model", "") or self.config.llm.model),
                    ),
                },
            )
        call_id = str(pending.get("tool_call_id") or "")
        answer_text = str(answer or "").strip() or "(no answer given)"
        if call_id and self._pending_call_exists(call_id):
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "ask_user",
                    "content": answer_text,
                }
            )
        else:
            # The assistant turn that asked the question is missing (corrupt
            # or hand-edited session).  A tool message without its call would
            # be rejected, so treat the answer as a new request instead.
            self.messages.append({"role": "user", "content": answer_text})
        self.session.pending_question = None
        self.session.status = "running"
        self._sync()
        return self._loop()

    def _pending_call_exists(self, call_id: str) -> bool:
        """True when the latest assistant turn contains the pending tool call."""
        for message in reversed(self.messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and str(call.get("id") or "") == call_id:
                    return True
            return False
        return False

    def cancel(self) -> None:
        self._cancel.set()
        self._status("cancelling")

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- the loop -----------------------------------------------------------
    def _loop(self) -> LoopResult:
        self._repair_tool_pairing()
        final = ""
        status = "done"
        error: str | None = None
        signatures: list[str] = []

        while True:
            if self._cancel.is_set():
                status = "cancelled"
                final = final or "Stopped at your request."
                break
            if self.steps >= self.max_steps:
                status = "max_steps"
                final = (
                    final
                    or f"Stopped after {self.max_steps} steps without a finish call. "
                    "The task may be incomplete."
                )
                break

            self._compact_if_needed()
            self.steps += 1
            self._status("thinking")
            self._emit("assistant_start", step=self.steps)
            try:
                result = self._chat()
            except Exception as exc:  # noqa: BLE001
                # The surface started an assistant turn; close it with whatever
                # was streamed so it is not left showing a stale typing state.
                self._emit(
                    "assistant_end",
                    content="".join(self._stream_parts),
                    step=self.steps,
                    tool_calls=[],
                )
                if self._cancel.is_set():
                    status = "cancelled"
                    final = "Stopped at your request."
                    break
                status = "error"
                error = str(exc)
                final = f"Model request failed: {exc}"
                self._emit("error", message=str(exc))
                break

            content = result.content or ""
            calls = list(result.tool_calls)

            # Local models may ignore native tool calling; accept one JSON action.
            if not calls and content.strip():
                fallback = self._parse_fallback(content)
                if fallback is not None:
                    calls = [fallback]
                else:
                    answer = self._fallback_final(content)
                    if answer is not None:
                        content = answer

            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                assistant_message["tool_calls"] = [call.to_message() for call in calls]
            self.messages.append(assistant_message)
            self._emit(
                "assistant_end",
                content=content,
                step=self.steps,
                tool_calls=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in calls
                ],
            )

            if self._cancel.is_set():
                # A stop during the model call wins over whatever the truncated
                # stream produced, and those tools must not run.
                for call in calls:
                    self._append_tool(call, "Not executed: the run was cancelled.", ok=False)
                final = "Stopped at your request."
                status = "cancelled"
                self._sync()
                break

            if not calls:
                final = content.strip() or "(the model returned an empty response)"
                self._sync()
                status = "done" if content.strip() else "empty"
                break

            signature = self._signature(calls)
            signatures.append(signature)
            if len(signatures) >= 4 and len(set(signatures[-4:])) == 1:
                final = (
                    "Stopped because the agent repeated the same tool call four times. "
                    "This usually means the tool failed in a way it kept retrying."
                )
                status = "stalled"
                # Every tool call needs a result or the next request is invalid.
                for call in calls:
                    self._append_tool(
                        call, "Not executed: repeated tool call detected.", ok=False
                    )
                self._emit("error", message=final)
                self._sync()
                break

            finished: FinishTask | None = None
            pending: AskUser | None = None
            pending_call_id = ""
            for index, call in enumerate(calls):
                if self._cancel.is_set():
                    # A stop that lands while an earlier tool is still running
                    # must keep the rest of the batch from executing.  They
                    # still need results, or the assistant turn is invalid.
                    for rest in calls[index:]:
                        self._append_tool(
                            rest, "Not executed: the run was cancelled.", ok=False
                        )
                    break
                self._emit("tool_start", id=call.id, name=call.name, args=call.arguments)
                started = time.monotonic()
                output = ""
                ok = True
                if call.argument_error:
                    ok = False
                    output = f"ERROR: {call.argument_error}"
                else:
                    try:
                        output = self.registry.execute(call.name, call.arguments, call.id)
                    except AskUser as ask:
                        pending = ask
                        pending_call_id = call.id
                        for rest in calls[index + 1 :]:
                            self._append_tool(
                                rest,
                                "Not executed: waiting for the user's answer.",
                                ok=False,
                            )
                        break
                    except FinishTask as finish:
                        finished = finish
                        output = "Task marked complete."
                    except (ToolError, OSError, ValueError) as exc:
                        ok = False
                        output = f"ERROR: {exc}"
                    except Exception as exc:  # noqa: BLE001 - report, never crash the loop
                        ok = False
                        output = f"ERROR: {type(exc).__name__}: {exc}"
                clipped = self._clip_tool_output(call, str(output))
                self._append_tool(call, clipped, ok=ok)
                self._emit(
                    "tool_end",
                    id=call.id,
                    name=call.name,
                    ok=ok,
                    content=clipped,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                if finished is not None:
                    for rest in calls[index + 1 :]:
                        self._append_tool(rest, "Not executed: task already finished.", ok=False)
                    break

            if pending is not None:
                pending_id = f"q_{int(time.time())}_{self.steps}"
                self.session.pending_question = {
                    "id": pending_id,
                    "tool_call_id": pending_call_id,
                    "question": pending.question,
                    "options": pending.options,
                }
                self.session.status = "awaiting_input"
                self._sync()
                self._emit(
                    "question",
                    id=pending_id,
                    question=pending.question,
                    options=pending.options,
                )
                return LoopResult(
                    status="awaiting_input",
                    content="",
                    steps=self.steps,
                    tokens=self.tokens,
                    session_id=self.session.session_id,
                    question={
                        "id": pending_id,
                        "question": pending.question,
                        "options": pending.options,
                    },
                    todos=self.registry.todos,
                    files_changed=sorted(set(self._files)),
                )

            if finished is not None:
                final = finished.summary
                status = "done"
                self._sync()
                break

            self._sync()

        self.session.status = "done" if status == "done" else status
        self.session.pending_question = None
        self._sync()
        self._emit(
            "done",
            content=final,
            status=status,
            steps=self.steps,
            tokens=self.tokens,
            error=error,
        )
        return LoopResult(
            status=status,
            content=final,
            steps=self.steps,
            tokens=self.tokens,
            session_id=self.session.session_id,
            error=error,
            todos=self.registry.todos,
            files_changed=sorted(set(self._files)),
        )

    # -- one model turn -----------------------------------------------------
    def _chat(self) -> ChatResult:
        tools = self.registry.specs()
        on_delta = self._on_delta if bool(getattr(self.config.llm, "stream", True)) else None
        self._stream_parts = []

        stop = threading.Event()
        started = time.monotonic()

        def heartbeat() -> None:
            while not stop.wait(10.0):
                self._emit("heartbeat", elapsed=int(time.monotonic() - started))

        beat = threading.Thread(target=heartbeat, name="filament-heartbeat", daemon=True)
        if self.sink:
            beat.start()
        try:
            try:
                return self.client.chat(
                    self.messages,
                    tools=tools,
                    on_delta=on_delta,
                    cancel=lambda: self._cancel.is_set(),
                )
            except Exception:
                # A retry after deltas would duplicate half an answer on screen,
                # so only transport failures before the first token are retried.
                if self._cancel.is_set() or self._stream_parts:
                    raise
                # One retry absorbs a dropped keep-alive or a flaky local server.
                time.sleep(0.5)
                return self.client.chat(
                    self.messages,
                    tools=tools,
                    on_delta=on_delta,
                    cancel=lambda: self._cancel.is_set(),
                )
        finally:
            stop.set()
            if self.sink and beat.is_alive():
                beat.join(timeout=1.0)

    def _on_delta(self, text: str) -> None:
        if text:
            self._stream_parts.append(text)
            self._emit("delta", text=text)

    def _parse_action(self, content: str) -> dict[str, Any] | None:
        """Decode the single JSON object a fallback reply may contain."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        if not text.startswith("{"):
            return None
        try:
            data, _ = json.JSONDecoder().raw_decode(text)
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _parse_fallback(self, content: str) -> ToolCall | None:
        """Accept a single JSON action from models without native tool calling."""
        data = self._parse_action(content)
        if data is None:
            return None
        if "final" in data and not any(key in data for key in ("tool", "name", "action")):
            return None
        action = str(data.get("action") or "").lower()
        name = data.get("tool") or data.get("name")
        if action and action not in {"tool", "call"}:
            return None
        if not isinstance(name, str) or not name.strip():
            return None
        # A plain JSON answer must not be mistaken for a tool call. Require an
        # explicit arguments object, which the protocol asks for.
        if "arguments" not in data and "args" not in data:
            return None
        arguments = data.get("arguments") or data.get("args") or {}
        if not isinstance(arguments, dict):
            return None
        return ToolCall(id=f"call_{int(time.time() * 1000)}", name=name.strip(), arguments=arguments)

    def _fallback_final(self, content: str) -> str | None:
        """Return the answer carried by a ``{"final": ...}`` reply.

        The system prompt tells models without native tool calling to answer
        that way, so the JSON envelope must be unwrapped: showing the user raw
        ``{"final": "..."}`` JSON reads like a broken agent.
        """
        data = self._parse_action(content)
        if data is None:
            return None
        if any(key in data for key in ("tool", "name", "action")):
            return None
        value = data.get("final")
        return value if isinstance(value, str) else None

    @staticmethod
    def _signature(calls: list[ToolCall]) -> str:
        return json.dumps(
            [(call.name, call.arguments) for call in calls],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    # -- message plumbing ---------------------------------------------------
    def _append_tool(self, call: ToolCall, output: str, ok: bool = True) -> None:
        self._append_tool_result(call.id, call.name, output, ok)

    def _append_tool_result(
        self,
        tool_call_id: str,
        name: str,
        output: str,
        ok: bool = True,
    ) -> None:
        message: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": str(tool_call_id),
            "name": str(name),
            "content": output,
        }
        if not ok:
            message["is_error"] = True
        self.messages.append(message)

    def _repair_tool_pairing(self) -> int:
        """Give every assistant tool call a result before the next model turn.

        A run interrupted during a tool (Ctrl+C, process kill) or a session that
        stopped on a question can leave an assistant ``tool_calls`` message with
        no matching ``tool`` message.  Providers reject that sequence, so the
        missing results are inserted where the conversation resumes.

        The reverse case is just as invalid: a ``tool`` message whose call is
        gone (a truncated or hand-edited session) is rejected outright, so it is
        dropped.  Duplicate results for one call are dropped too.
        """
        repaired = 0
        index = 0
        while index < len(self.messages):
            message = self.messages[index]
            if not isinstance(message, dict):
                index += 1
                continue
            if message.get("role") == "tool":
                # No preceding assistant tool call owns this result.
                self.messages.pop(index)
                repaired += 1
                continue
            if message.get("role") != "assistant":
                index += 1
                continue
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                index += 1
                continue
            calls: list[dict[str, Any]] = []
            for call in raw_calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                if not isinstance(name, str) or not name.strip():
                    # Every provider adapter drops a call with no name.  Pairing
                    # a result for it would put a tool message on the wire with
                    # no call in front of it, which the provider rejects.
                    continue
                calls.append(call)
            if len(calls) != len(raw_calls):
                repaired += 1
                if calls:
                    message["tool_calls"] = calls
                else:
                    message.pop("tool_calls", None)
            wanted: list[str] = []
            for position, call in enumerate(calls, 1):
                call_id = str(call.get("id") or "")
                if not call_id:
                    call_id = f"call_recovered_{index}_{position}"
                    call["id"] = call_id
                wanted.append(call_id)
            cursor = index + 1
            answered: set[str] = set()
            while cursor < len(self.messages):
                candidate = self.messages[cursor]
                if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                    break
                call_id = str(candidate.get("tool_call_id") or "")
                if call_id in wanted and call_id not in answered:
                    answered.add(call_id)
                    cursor += 1
                    continue
                # A result for an earlier turn, a duplicate, or an id this
                # turn never issued: providers reject those.
                self.messages.pop(cursor)
                repaired += 1
            for position, call in enumerate(calls, 1):
                call_id = wanted[position - 1]
                if call_id in answered:
                    continue
                name = str(call.get("name") or "tool")
                content = (
                    "Not executed: the user did not answer this question."
                    if name == "ask_user"
                    else f"Not executed: the previous turn was interrupted before {name} returned a result."
                )
                self.messages.insert(
                    cursor,
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": content,
                        "is_error": True,
                    },
                )
                cursor += 1
                answered.add(call_id)
                repaired += 1
            index = cursor
        return repaired

    def _clip_tool_output(self, call: ToolCall, output: str) -> str:
        limit = max(256, int(self.config.tool_output_limit))
        if len(output) <= limit:
            return output
        saved = None
        try:
            directory = self.workspace / ".filament" / "runs"
            directory.mkdir(parents=True, exist_ok=True)
            # Tool call ids come from the provider, so they are untrusted path
            # input.  Keep the log inside .filament/runs even for an id that
            # contains separators or "..".
            safe_id = "".join(
                ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "_"
                for ch in str(call.id)
            )[:40] or "call"
            path = directory / f"tool-{safe_id}-{int(time.time())}.log"
            path.write_text(output, encoding="utf-8", errors="replace")
            saved = path.relative_to(self.workspace).as_posix()
        except (OSError, ValueError):
            saved = None
        marker = f"\n... [tool output truncated: {len(output)} chars total] ...\n"
        head = (limit - len(marker)) // 2
        tail = limit - len(marker) - head
        clipped = output[:head] + marker + output[-tail:]
        if saved:
            clipped += f"\n[full output saved: {saved}]"
        return clipped

    def _compact_if_needed(self) -> None:
        self.tokens = estimate_tokens(self.messages, self.config.chars_per_token)
        threshold = int(self.budget_tokens * self.config.compact_threshold)
        if self.tokens <= threshold:
            self._emit_context()
            return
        messages, changed, stats = compact_messages(
            self.client if self.config.compact_with_llm else None,
            self.messages,
            self.budget_tokens,
            self.config.chars_per_token,
            keep_recent=self.config.keep_recent_messages,
            threshold=self.config.compact_threshold,
            use_llm=self.config.compact_with_llm and self.depth == 0,
        )
        if changed:
            self.messages = messages
            self.compactions += 1
            self.tokens = estimate_tokens(self.messages, self.config.chars_per_token)
            self._emit("compaction", **stats)
            self._sync()
        self._emit_context()

    def _sync(self) -> None:
        with self._lock:
            self.session.messages = self.messages
            self.session.todos = list(self.registry.todos)
            self.session.files_changed = sorted(set(self._files))
            self.session.usage = {
                # Telemetry accumulates over the session; the step *limit* is
                # applied to self.steps, which resets every turn.
                "steps": self._prior_steps + self.steps,
                "tokens": self.tokens,
                "compactions": self.compactions,
            }
            try:
                save_session(self.session, self.workspace)
            except (OSError, ValueError):
                pass

    def _on_file_change(self, path: str) -> None:
        with self._lock:
            if path not in self._files:
                self._files.append(path)

    def _on_todos(self, todos: list[dict[str, Any]]) -> None:
        self._emit("todos", todos=todos)

    # -- subagents ----------------------------------------------------------
    def _spawn_agent(self, task: str, system: str | None, call_id: str) -> dict[str, Any]:
        if self.depth >= self.config.max_depth:
            raise ToolError("subagent depth limit reached")
        child = create_session(
            self.workspace,
            mode="plan" if self.read_only else "build",
            parent_id=self.session.session_id,
            kind="subagent",
            title=(task.strip().splitlines() or [""])[0][:80],
        )
        self.session.child_sessions.append(child.session_id)
        self._emit(
            "subagent_start",
            id=call_id,
            task=task,
            child_session_id=child.session_id,
            parent_session_id=self.session.session_id,
        )
        loop = AgentLoop(
            config=self.config,
            workspace=self.workspace,
            session=child,
            client=self.client,
            event_sink=self.sink,
            depth=self.depth + 1,
            read_only=self.read_only,
            max_steps=self.config.subagent_max_steps,
            budget_tokens=self.config.subagent_max_tokens,
            subagent_id=call_id,
            cancel_event=self._cancel,
            system_extra=system or "",
        )
        try:
            result = loop.run(task)
        except Exception as exc:  # noqa: BLE001 - a failed child must not kill the parent
            self._emit("subagent_end", id=call_id, child_session_id=child.session_id, status="error", steps=0, tokens=0, content=str(exc))
            raise ToolError(f"subagent failed: {exc}") from None
        self._emit(
            "subagent_end",
            id=call_id,
            child_session_id=child.session_id,
            status=result.status,
            steps=result.steps,
            tokens=result.tokens,
            content=result.content,
        )
        # The parent's change list should reflect work the child did on its behalf.
        for changed in result.files_changed:
            self._on_file_change(changed)
        return {
            "session_id": child.session_id,
            "status": result.status,
            "steps": result.steps,
            "tokens": result.tokens,
            "content": result.content,
        }
