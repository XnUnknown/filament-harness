"""Configuration loading for the filament harness."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    kind: str = "auto"  # auto, mock, ollama
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    use_ollama_env: bool = False

    def validate(self) -> None:
        if self.kind not in {"auto", "mock", "ollama"}:
            raise ValueError(f"unknown llm kind: {self.kind}")
        if not self.model.strip():
            raise ValueError("llm model must not be empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("llm base_url must be http(s)")


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
    command_allowlist: list[str] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def validate(self) -> "HarnessConfig":
        if self.max_tokens < 1024:
            raise ValueError("max_tokens must be at least 1024")
        if self.chars_per_token < 1:
            raise ValueError("chars_per_token must be positive")
        if not (0.0 <= self.budget_margin < 0.5):
            raise ValueError("budget_margin must be between 0 and 0.5")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.max_tasks < 1:
            raise ValueError("max_tasks must be at least 1")
        if self.timeout_shell < 1:
            raise ValueError("timeout_shell must be at least 1")
        if self.max_shell_output < 256:
            raise ValueError("max_shell_output must be at least 256")
        self.llm.validate()
        return self

    @property
    def effective_max_tokens(self) -> int:
        return int(self.max_tokens * (1.0 - self.budget_margin))


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
    llm_data = data.get("llm", {})
    llm = LLMConfig(**{k: v for k, v in llm_data.items() if k in LLMConfig.__dataclass_fields__})
    cfg_data = {k: v for k, v in data.items() if k != "llm" and k in HarnessConfig.__dataclass_fields__}
    return HarnessConfig(llm=llm, **cfg_data)


def load_config(path: str | Path | None = None) -> HarnessConfig:
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
        "FILAMENT_CLIENT": ("llm.kind", str),
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

    if os.environ.get("FILAMENT_USE_OLLAMA") == "1":
        data.setdefault("llm", {})["kind"] = "ollama"

    cfg = _from_dict(data)
    cfg.validate()
    return cfg
