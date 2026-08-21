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
ACCENT = "#67dfc2"

CHROME = "#4d7cff"  # hyperplex.org accent
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


def esc(text: str) -> str:
    """
    Escape text for interpolation into rich markup.

    Every helper below runs its text arguments through this: messages
    routinely carry file paths, model ids and scraped document text, and
    a stray `[/]` in those must never reach the markup parser.
    """
    from rich.markup import escape

    return escape(text)


def banner(tagline: str = "") -> None:
    """`{sigil} {tool} · ⟡ hyperplex` over a thin chrome rule."""
    c = console()
    if c is None:
        print(f"{TOOL} — hyperplex", file=sys.stderr)
        return
    line = f"[bold {ACCENT}]{SIGIL} {TOOL}[/]  [dim]·[/]  [{CHROME}]{MARK} hyperplex[/]"
    if tagline:
        line += f"  [dim]· {esc(tagline)}[/]"
    c.print(line)
    c.print(f"[{CHROME}]{'─' * min(46, c.width)}[/]")


def step(message: str) -> None:
    """Announce a stage: `◆ message` in the tool accent."""
    c = console()
    if c is None:
        print(f"* {message}", file=sys.stderr)
    else:
        c.print(f"[bold {ACCENT}]◆[/] {esc(message)}")


def say(message: str) -> None:
    c = console()
    if c is None:
        print(message, file=sys.stderr)
    else:
        c.print(f"[dim]{esc(message)}[/]")


def ok(message: str) -> None:
    c = console()
    if c is None:
        print(f"ok: {message}", file=sys.stderr)
    else:
        c.print(f"[green]✓[/] {esc(message)}")


def warn(message: str) -> None:
    c = console()
    if c is None:
        print(f"warning: {message}", file=sys.stderr)
    else:
        c.print(f"[yellow]![/] {esc(message)}")


def fail(message: str) -> None:
    c = console()
    if c is None:
        print(f"error: {message}", file=sys.stderr)
    else:
        c.print(f"[bold red]✗[/] {esc(message)}")


def _read(prompt_markup: str, prompt_plain: str) -> str:
    c = console()
    if c is None:
        return input(prompt_plain).strip()
    return c.input(prompt_markup).strip()


def ask(prompt: str, default: Optional[str] = None) -> str:
    """Ask for a free-form value; empty input returns the default."""
    suffix = f" [dim]\\[{esc(default)}][/]" if default else ""
    plain_suffix = f" [{default}]" if default else ""
    while True:
        answer = _read(
            f"[bold {ACCENT}]»[/] {esc(prompt)}{suffix} ",
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
            f"[bold {ACCENT}]»[/] {esc(prompt)} [dim]\\[{hint}][/] ",
            f"> {prompt} [{hint}] ",
        ).lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def _menu_step(key: str, index: int, count: int) -> tuple:
    """One keypress in a menu: (new highlight index, confirmed?).

    ↑/k and ↓/j move (wrapping), a digit jumps to that option, and
    Enter confirms — so the old "type 2, Enter" habit still works.
    """
    if key in ("\r", "\n"):
        return index, True
    if key in ("up", "k"):
        return (index - 1) % count, False
    if key in ("down", "j"):
        return (index + 1) % count, False
    if key.isdigit() and 1 <= int(key) <= count:
        return int(key) - 1, False
    return index, False


def _read_menu_key(fd: int) -> str:
    """One keypress, decoding arrow escape sequences to "up"/"down"."""
    import select

    data = os.read(fd, 1)
    if data != b"\x1b":
        return data.decode("utf-8", "ignore")
    seq = b""
    while len(seq) < 2 and select.select([fd], [], [], 0.05)[0]:
        seq += os.read(fd, 1)
    if seq == b"[A":
        return "up"
    if seq == b"[B":
        return "down"
    return ""


def _choose_with_keys(prompt: str, options: List[str], default: Optional[int]) -> Optional[int]:
    """In-place ↑/↓ (or j/k) menu; None when key input is unavailable.

    Repaints the highlighted line with cursor movements rather than
    taking over the screen, and collapses to a one-line record of the
    choice — so transcripts stay readable and piped runs (no tty) fall
    back to the numbered prompt.
    """
    c = console()
    if c is None or not sys.stdin.isatty():
        return None
    try:
        import termios
        import tty
    except ImportError:  # not a POSIX terminal (e.g. Windows)
        return None

    def draw(index: int) -> None:
        for i, option in enumerate(options):
            sys.stderr.write("\x1b[2K")
            if i == index:
                c.print(f"[bold {ACCENT}]❯ {i + 1}) {esc(option)}[/]", no_wrap=True)
            else:
                c.print(f"  [dim]{i + 1})[/] {esc(option)}", no_wrap=True)
        sys.stderr.write("\x1b[2K")
        c.print("[dim]↑/↓ · j/k · number · enter[/]", no_wrap=True)

    index = default if default is not None else 0
    c.print(f"\n{esc(prompt)}")
    draw(index)
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        # cbreak keeps ISIG, so Ctrl-C still interrupts. TCSANOW: the
        # default TCSAFLUSH would wait for our own menu output to drain,
        # which deadlocks when nothing is reading the terminal (ptys in
        # tests) -- and only input flags change, so there is nothing to
        # protect by draining.
        tty.setcbreak(fd, termios.TCSANOW)
        while True:
            index, done = _menu_step(_read_menu_key(fd), index, len(options))
            if done:
                break
            sys.stderr.write(f"\x1b[{len(options) + 1}F")
            draw(index)
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)
    sys.stderr.write(f"\x1b[{len(options) + 2}F\x1b[0J")
    c.print(f"[bold {ACCENT}]»[/] {esc(prompt)} [dim]· {esc(options[index])}[/]")
    return index


def choose(prompt: str, options: List[str], default: Optional[int] = None) -> int:
    """Pick one option: ↑/↓ or j/k and Enter on a tty, numbered input otherwise.

    Returns the 0-based index.
    """
    try:
        picked = _choose_with_keys(prompt, options, default)
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        picked = None  # odd terminal: fall back to the numbered prompt
    if picked is not None:
        return picked

    c = console()
    if c is None:
        print(f"\n{prompt}", file=sys.stderr)
        for i, option in enumerate(options):
            marker = "*" if default is not None and i == default else " "
            print(f" {marker} {i + 1}) {option}", file=sys.stderr)
    else:
        c.print(f"\n{esc(prompt)}")
        for i, option in enumerate(options):
            is_default = default is not None and i == default
            marker = f"[bold {ACCENT}]{i + 1}[/]"
            label = f"[bold]{esc(option)}[/]" if is_default else esc(option)
            suffix = " [dim](default)[/]" if is_default else ""
            c.print(f"  {marker}) {label}{suffix}")
    while True:
        raw = _read(f"[bold {ACCENT}]»[/] ", "> ")
        if not raw and default is not None:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        matches = [i for i, o in enumerate(options) if o.lower().startswith(raw.lower())]
        if len(matches) == 1:
            return matches[0]
        warn(f"enter a number between 1 and {len(options)}")
