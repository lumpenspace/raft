"""
Minimal interactive prompt helpers used by the interactive CLI modes.

No external dependencies: plain stdin/stdout, suitable for piping raft
around other interactive tools (ariadne, opbdh) that take over the tty.
"""

import os
import sys
from typing import List, Optional


def ask(prompt: str, default: Optional[str] = None) -> str:
    """
    Ask for a free-form value.

    Args:
        prompt (str): The question to display.
        default (Optional[str]): Value returned on empty input.

    Returns:
        str: The stripped answer (or the default).
    """
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default


def confirm(prompt: str, default: bool = True) -> bool:
    """
    Ask a yes/no question.

    Args:
        prompt (str): The question to display.
        default (bool): Value returned on empty input.

    Returns:
        bool: True for yes, False for no.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def choose(prompt: str, options: List[str], default: Optional[int] = None) -> int:
    """
    Ask the user to pick one of several options.

    Args:
        prompt (str): The question to display.
        options (List[str]): Human-readable option labels.
        default (Optional[int]): 0-based index returned on empty input.

    Returns:
        int: The 0-based index of the chosen option.
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        marker = "*" if default is not None and i == default else " "
        print(f" {marker} {i + 1}) {option}")
    while True:
        raw = input("> ").strip()
        if not raw and default is not None:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"Please enter a number between 1 and {len(options)}.")


def ask_path(prompt: str, default: Optional[str] = None, must_exist: bool = True) -> str:
    """
    Ask for a filesystem path, optionally requiring it to exist.

    Args:
        prompt (str): The question to display.
        default (Optional[str]): Value returned on empty input.
        must_exist (bool): Re-prompt until the path exists.

    Returns:
        str: The expanded path.
    """
    while True:
        path = os.path.expanduser(ask(prompt, default))
        if not must_exist or os.path.exists(path):
            return path
        print(f"Path not found: {path}")


def bail(message: str) -> None:
    """Print an error message and exit with a non-zero status."""
    print(f"\nerror: {message}", file=sys.stderr)
    sys.exit(1)
