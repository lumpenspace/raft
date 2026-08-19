"""
Interactive prompt helpers for the interactive CLI modes.

Thin wrappers over the hyperplex chrome in `hx` (styled when stderr is a
tty, plain ASCII otherwise), so raft can sit in a pipeline of other
interactive tools without corrupting piped output.
"""

import os
import sys
from typing import List, Optional

from . import hx
from .hx import ask, choose, confirm  # noqa: F401  (re-exported API)


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
        hx.warn(f"path not found: {path}")


def bail(message: str) -> None:
    """Print an error message and exit with a non-zero status."""
    hx.fail(message)
    sys.exit(1)


__all__: List[str] = ["ask", "confirm", "choose", "ask_path", "bail"]
