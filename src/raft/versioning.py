"""Installed and published raft-ft version reporting."""

from packaging.version import InvalidVersion, Version
import requests

from . import __version__


DISTRIBUTION = "raft-ft"
PYPI_JSON_URL = f"https://pypi.org/pypi/{DISTRIBUTION}/json"


class VersionCheckError(RuntimeError):
    """The latest published version could not be determined."""


def latest_pypi_version() -> str:
    """Return the latest raft-ft version published on PyPI."""
    try:
        response = requests.get(
            PYPI_JSON_URL,
            headers={"Accept": "application/json", "User-Agent": "raft-ft version"},
            timeout=10,
        )
        response.raise_for_status()
        value = response.json()["info"]["version"]
        Version(value)
    except (requests.RequestException, KeyError, TypeError, InvalidVersion, ValueError) as exc:
        raise VersionCheckError(f"could not check PyPI: {exc}") from exc
    return value


def report_versions() -> int:
    """Print the installed and latest published versions without changing anything."""
    print(f"Installed: {DISTRIBUTION} {__version__}")
    try:
        latest = latest_pypi_version()
    except VersionCheckError as exc:
        print(f"Latest:    unavailable ({exc})")
        return 1

    print(f"Latest:    {DISTRIBUTION} {latest} (PyPI)")
    try:
        current_version = Version(__version__)
        latest_version = Version(latest)
    except InvalidVersion as exc:
        print(f"Status:    unable to compare versions ({exc})")
        return 1

    if current_version < latest_version:
        status = "update available"
    elif current_version > latest_version:
        status = "installed version is newer than PyPI"
    else:
        status = "up to date"
    print(f"Status:    {status}")
    return 0
