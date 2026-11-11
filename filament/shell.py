"""Command-line splitting shared by every shell-running tool surface.

Both :class:`filament.tools.Toolset` and :class:`filament.agent_tools.Toolbox`
run commands with ``shell=False`` when ``use_shell`` is disabled. They must
tokenize the command the same way, so the platform handling lives here once.
"""
from __future__ import annotations

import os
import shlex


def split_command(command: str) -> list[str]:
    """Split a command line into argv without asking a shell to interpret it."""
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    if os.name != "nt":
        argv = shlex.split(command)
    else:
        # POSIX-mode shlex treats backslashes as escapes, which mangles Windows
        # paths such as C:\\Users\\name. Non-POSIX mode keeps them but retains
        # one layer of surrounding quotes, so strip that layer afterwards.
        argv = []
        for part in shlex.split(command, posix=False):
            if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"}:
                part = part[1:-1]
            argv.append(part)
    if not argv or not argv[0].strip():
        raise ValueError("command has no program name")
    return argv


def prepare_command(command: str, use_shell: bool) -> str | list[str]:
    """Return the form ``subprocess.run`` should receive for this command."""
    if use_shell:
        return command
    return split_command(command)
