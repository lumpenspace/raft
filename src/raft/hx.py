"""hyperplex terminal chrome for raft.

Vendored in each hyperplex tool (raft, ariadne, opbdh) with only the
TOOL/SIGIL/ACCENT constants changing; see docs/HYPERPLEX.md for the spec.

All chrome goes to stderr so piped stdout stays machine-clean, and
everything degrades to plain ASCII when stderr is not a tty or NO_COLOR /
HYPERPLEX_PLAIN is set.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

TOOL = "raft"
SIGIL = "≋"
ACCENT = "#22d3ee"

VIOLET = "#a78bfa"
MARK = "⟡"


def _plain() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("HYPERPLEX_PLAIN"):
        return True
    try:
        return not sys.stderr.isatty()
    except Exception:
        return True


_console = None


def console():
    """The stderr rich Console, or None in plain mode."""
    global _console
    if _plain():
        return None
    if _console is None:
        try:
            from rich.console import Console

            _console = Console(stderr=True, highlight=False)
        except ImportError:
            return None
    return _console


def banner(tagline: str = "") -> None:
    """`≋ raft · ⟡ hyperplex` over a thin violet rule."""
    c = console()
    if c is None:
        print(f"{TOOL} — hyperplex", file=sys.stderr)
        return
    line = f"[bold {ACCENT}]{SIGIL} {TOOL}[/]  [dim]·[/]  [{VIOLET}]{MARK} hyperplex[/]"
    if tagline:
        line += f"  [dim]· {tagline}[/]"
    c.print(line)
    c.print(f"[{VIOLET}]{'─' * min(46, c.width)}[/]")


def step(message: str) -> None:
    """Announce a stage: `◆ message` in the tool accent."""
    c = console()
    if c is None:
        print(f"* {message}", file=sys.stderr)
    else:
        c.print(f"[bold {ACCENT}]◆[/] {message}")


def say(message: str) -> None:
    c = console()
    if c is None:
        print(message, file=sys.stderr)
    else:
        c.print(f"[dim]{message}[/]")


def ok(message: str) -> None:
    c = console()
    if c is None:
        print(f"ok: {message}", file=sys.stderr)
    else:
        c.print(f"[green]✓[/] {message}")


def warn(message: str) -> None:
    c = console()
    if c is None:
        print(f"warning: {message}", file=sys.stderr)
    else:
        c.print(f"[yellow]![/] {message}")


def fail(message: str) -> None:
    c = console()
    if c is None:
        print(f"error: {message}", file=sys.stderr)
    else:
        c.print(f"[bold red]✗[/] {message}")


def _read(prompt_markup: str, prompt_plain: str) -> str:
    c = console()
    if c is None:
        return input(prompt_plain).strip()
    return c.input(prompt_markup).strip()


def ask(prompt: str, default: Optional[str] = None) -> str:
    """Ask for a free-form value; empty input returns the default."""
    suffix = f" [dim]\\[{default}][/]" if default else ""
    plain_suffix = f" [{default}]" if default else ""
    while True:
        answer = _read(
            f"[bold {ACCENT}]»[/] {prompt}{suffix} ",
            f"> {prompt}{plain_suffix} ",
        )
        if answer:
            return answer
        if default is not None:
            return default


def confirm(prompt: str, default: bool = True) -> bool:
    """Yes/no question."""
    hint = "Y/n" if default else "y/N"
    while True:
        answer = _read(
            f"[bold {ACCENT}]»[/] {prompt} [dim]\\[{hint}][/] ",
            f"> {prompt} [{hint}] ",
        ).lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def choose(prompt: str, options: List[str], default: Optional[int] = None) -> int:
    """Pick one option by number; returns the 0-based index."""
    c = console()
    if c is None:
        print(f"\n{prompt}", file=sys.stderr)
        for i, option in enumerate(options):
            marker = "*" if default is not None and i == default else " "
            print(f" {marker} {i + 1}) {option}", file=sys.stderr)
    else:
        c.print(f"\n{prompt}")
        for i, option in enumerate(options):
            is_default = default is not None and i == default
            marker = f"[bold {ACCENT}]{i + 1}[/]"
            label = f"[bold]{option}[/]" if is_default else option
            suffix = " [dim](default)[/]" if is_default else ""
            c.print(f"  {marker}) {label}{suffix}")
    while True:
        raw = _read(f"[bold {ACCENT}]»[/] ", "> ")
        if not raw and default is not None:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        warn(f"enter a number between 1 and {len(options)}")
