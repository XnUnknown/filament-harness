"""Configuration loading for the filament harness."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from .llm import _is_local_url


def _type_name(value: Any) -> str:
    return type(value).__name__


def _preview(value: Any, limit: int = 40) -> str:
    """Short repr so error text stays readable and never dumps a whole value."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def coerce_str(value: Any, name: str) -> str:
    """Accept a non-empty string, rejecting numbers and other scalars."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {_type_name(value)} {_preview(value)}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def coerce_optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or unset, got {_type_name(value)} {_preview(value)}")
    return value


def coerce_int(value: Any, name: str, minimum: int | None = None) -> int:
    """Accept an integer (or a whole float), rejecting bools, strings, fractions."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {_type_name(value)} {_preview(value)}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def coerce_number(value: Any, name: str) -> float:
    """Accept an int or float, rejecting bools and strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {_type_name(value)} {_preview(value)}")
    return float(value)


def coerce_bool(value: Any, name: str) -> bool:
    """Accept only a real boolean; a quoted "false" must never act like true."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {_type_name(value)} {_preview(value)}")
    return value


def coerce_str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings, got {_type_name(value)} {_preview(value)}")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} entries must be non-empty strings, got {_type_name(item)} {_preview(item)}")
        result.append(item)
    return result


class _ConfigIssues:
    """Collect per-field failures so one load reports every bad key at once."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def run(self, check: Callable[[], Any]) -> None:
        try:
            check()
        except ValueError as exc:
            self.messages.append(str(exc))

    def raise_if_any(self, prefix: str = "") -> None:
        if not self.messages:
            return
        detail = "; ".join(self.messages)
        raise ValueError(f"{prefix}: {detail}" if prefix else detail)


# Integer config keys and their minimum accepted value.
_INT_FIELDS = {
    "max_tokens": 1024,
    "chars_per_token": 1,
    "workers": 1,
    "max_tasks": 1,
    "timeout_shell": 1,
    "max_shell_output": 256,
    "max_steps": 1,
    "subagent_max_steps": 1,
    "subagent_max_tokens": 1024,
    "tool_output_limit": 256,
    "keep_recent_messages": 2,
    "max_depth": 0,
}
_BOOL_FIELDS = (
    "allow_outside_target",
    "atomic_writes",
    "use_shell",
    "offload_shell_output",
    "compact_with_llm",
)
_LLM_KINDS = ("auto", "mock", "ollama", "openai", "anthropic")


@dataclass
class LLMConfig:
    kind: str = "auto"  # auto, mock, ollama, openai, anthropic
    model: str = "llama3.2"
    base_url: str | None = None
    api_key: str | None = None
    timeout: int = 120
    reasoning_effort: str | None = None
    use_ollama_env: bool = False
    stream: bool = True
    max_output_tokens: int = 4096

    def validate(self) -> None:
        issues = _ConfigIssues()
        issues.run(self._coerce_types)
        issues.run(self._check_timeout)
        issues.run(self._check_output_tokens)
        issues.raise_if_any()
        self._check_semantics()

    def _coerce_types(self) -> None:
        # Field order decides which wrong llm value is named first.
        self.kind = coerce_str(self.kind, "llm.kind")
        self.model = coerce_str(self.model, "llm.model")
        self.base_url = coerce_optional_str(self.base_url, "llm.base_url")
        self.api_key = coerce_optional_str(self.api_key, "llm.api_key")
        self.reasoning_effort = coerce_optional_str(self.reasoning_effort, "llm.reasoning_effort")
        self.use_ollama_env = coerce_bool(self.use_ollama_env, "llm.use_ollama_env")
        self.stream = coerce_bool(self.stream, "llm.stream")

    def _check_output_tokens(self) -> None:
        self.max_output_tokens = coerce_int(self.max_output_tokens, "llm.max_output_tokens", 64)

    def _check_timeout(self) -> None:
        if self.timeout is None:
            self.timeout = 120
        timeout = coerce_timeout(self.timeout, 120, "llm timeout")
        if timeout < 1:
            raise ValueError("llm timeout must be at least 1 second")
        if timeout != int(timeout):
            raise ValueError("llm timeout must be a whole number of seconds")
        self.timeout = int(timeout)

    def _check_semantics(self) -> None:
        if self.kind not in _LLM_KINDS:
            raise ValueError(f"unknown llm kind: {self.kind}")
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("llm base_url must be http(s)")
        if self.reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("reasoning_effort must be low, medium, high, or unset")
        if self.kind in {"openai", "anthropic", "ollama"}:
            if self.kind == "ollama":
                key = self.api_key or os.environ.get("FILAMENT_API_KEY") or os.environ.get("OLLAMA_API_KEY")
                # Ollama defaults to a local server, so keys are optional there.
                url_for_local = self.base_url or os.environ.get("FILAMENT_OLLAMA_URL") or "http://localhost:11434"
                local = _is_local_url(url_for_local)
            else:
                key = self.api_key or os.environ.get("FILAMENT_API_KEY")
                # Local OpenAI-compatible servers may omit keys; require for hosted endpoints.
                local = bool(self.base_url) and _is_local_url(self.base_url)
            if not key and not local:
                raise ValueError(f"api_key is required for {self.kind} provider")


@dataclass
class HarnessConfig:
    max_tokens: int = 120_000
    chars_per_token: int = 4
    budget_margin: float = 0.05
    workers: int = 4
    max_tasks: int = 32
    timeout_shell: int = 60
    max_shell_output: int = 20_000
    allow_outside_target: bool = False
    atomic_writes: bool = True
    use_shell: bool = True
    offload_shell_output: bool = True
    # Agent loop
    max_steps: int = 40
    subagent_max_steps: int = 24
    subagent_max_tokens: int = 32_000
    tool_output_limit: int = 12_000
    keep_recent_messages: int = 8
    compact_threshold: float = 0.75
    compact_with_llm: bool = True
    max_depth: int = 1
    command_allowlist: list[str] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def validate(self) -> "HarnessConfig":
        """Coerce and type-check every field; report all bad keys in one error."""
        issues = _ConfigIssues()
        for name, minimum in _INT_FIELDS.items():
            issues.run(lambda name=name, minimum=minimum: self._set_int(name, minimum))
        for name in _BOOL_FIELDS:
            issues.run(lambda name=name: self._set_bool(name))
        issues.run(self._check_budget_margin)
        issues.run(self._check_compact_threshold)
        issues.run(self._apply_allowlist)
        issues.run(self.llm.validate)
        issues.raise_if_any("invalid configuration")
        return self

    def _set_int(self, name: str, minimum: int) -> None:
        setattr(self, name, coerce_int(getattr(self, name), name, minimum))

    def _set_bool(self, name: str) -> None:
        setattr(self, name, coerce_bool(getattr(self, name), name))

    def _apply_allowlist(self) -> None:
        self.command_allowlist = coerce_str_list(self.command_allowlist, "command_allowlist")

    def _check_budget_margin(self) -> None:
        margin = coerce_number(self.budget_margin, "budget_margin")
        if not (0.0 <= margin < 0.5):
            raise ValueError("budget_margin must be between 0 and 0.5")
        self.budget_margin = margin

    def _check_compact_threshold(self) -> None:
        threshold = coerce_number(self.compact_threshold, "compact_threshold")
        if not (0.1 <= threshold <= 0.95):
            raise ValueError("compact_threshold must be between 0.1 and 0.95")
        self.compact_threshold = threshold

    @property
    def effective_max_tokens(self) -> int:
        return int(self.max_tokens * (1.0 - self.budget_margin))


MAX_SHELL_TIMEOUT_SECONDS = 86_400
DEFAULT_READ_MAX_CHARS = 12_000


def coerce_timeout(value: Any, default: float, label: str = "timeout") -> float:
    """Validate a timeout in seconds coming from config or untrusted arguments."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number of seconds, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    if value > MAX_SHELL_TIMEOUT_SECONDS:
        raise ValueError(f"{label} must not exceed {MAX_SHELL_TIMEOUT_SECONDS} seconds")
    return float(value)


def coerce_char_limit(value: Any, default: int = DEFAULT_READ_MAX_CHARS) -> int:
    """Validate a character cap coming from untrusted tool arguments."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("max_chars must be an integer")
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise ValueError(f"max_chars must be an integer, got {value!r}") from None
    if not isinstance(value, int):
        raise ValueError(f"max_chars must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError("max_chars must be positive")
    return value


def coerce_max_fixes(value: Any, default: int | None = None) -> int | None:
    """Validate a review-fix cap coming from config or untrusted arguments."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("max_fixes must be an integer")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"max_fixes must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError("max_fixes must not be negative")
    return value


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _from_dict(data: dict[str, Any]) -> HarnessConfig:
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a table, got {type(data).__name__}")
    llm_data = data.get("llm", {})
    if llm_data is None:
        llm_data = {}
    if not isinstance(llm_data, dict):
        raise ValueError(f"llm must be a table, got {type(llm_data).__name__} {_preview(llm_data)}")
    llm = LLMConfig(**{k: v for k, v in llm_data.items() if k in LLMConfig.__dataclass_fields__})
    cfg_data = {k: v for k, v in data.items() if k != "llm" and k in HarnessConfig.__dataclass_fields__}
    return HarnessConfig(llm=llm, **cfg_data)


def load_config(path: str | Path | None = None) -> HarnessConfig:
    # Support a simple project-local .env file without adding dependencies.
    if path is None:
        dotenv_candidates = [Path(".env")]
    else:
        dotenv_candidates = [Path(path).parent / ".env"]
    for env_path in dotenv_candidates:
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)

    search_paths: list[Path] = []
    if path is not None:
        search_paths.append(Path(path))
    else:
        search_paths.extend([Path("filament.toml"), Path(".filament.toml")])

    data: dict[str, Any] = {}
    for p in search_paths:
        if p.exists():
            data = _merge_dict(data, _read_toml(p))
            # Also accept [tool.filament] from pyproject for compatibility
            if p.name == "pyproject.toml":
                tool = data.get("tool", {}).get("filament", {})
                if isinstance(tool, dict):
                    data = _merge_dict(data, tool)
            break

    # Environment overrides.
    env_map = {
        "FILAMENT_MAX_TOKENS": ("max_tokens", int),
        "FILAMENT_WORKERS": ("workers", int),
        "FILAMENT_TIMEOUT_SHELL": ("timeout_shell", int),
        "FILAMENT_MODEL": ("llm.model", str),
        "FILAMENT_OLLAMA_URL": ("llm.base_url", str),
        "FILAMENT_API_KEY": ("llm.api_key", str),
        "FILAMENT_CLIENT": ("llm.kind", str),
        "FILAMENT_TIMEOUT": ("llm.timeout", int),
        "FILAMENT_REASONING_EFFORT": ("llm.reasoning_effort", str),
    }
    for env_name, (field_path, cast) in env_map.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            converted = cast(value)
        except (TypeError, ValueError):
            continue
        if "." in field_path:
            section, key = field_path.split(".", 1)
            data.setdefault(section, {})[key] = converted
        else:
            data[field_path] = converted

    # Ollama cloud credentials may be provided under their native variable name.
    ollama_key = os.environ.get("OLLAMA_API_KEY")
    if ollama_key and not os.environ.get("FILAMENT_API_KEY") and not data.get("llm", {}).get("api_key"):
        current_kind = data.get("llm", {}).get("kind")
        if current_kind in (None, "auto", "ollama"):
            data.setdefault("llm", {})["api_key"] = ollama_key

    if os.environ.get("FILAMENT_USE_OLLAMA") == "1":
        data.setdefault("llm", {})["kind"] = "ollama"

    cfg = _from_dict(data)
    cfg.validate()
    return cfg
