"""Installed and published version reporting."""

from unittest.mock import Mock, patch

import pytest
import requests

from raft import __version__, cli, versioning


def test_version_flag_is_offline_and_works_outside_a_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["raft", "--version"])

    with patch("requests.get") as get, pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert capsys.readouterr().out == f"raft-ft {__version__}\n"
    get.assert_not_called()


def test_version_action_compares_with_pypi_outside_a_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["raft", "version"])

    with patch.object(versioning, "latest_pypi_version", return_value=__version__):
        cli.main()

    assert capsys.readouterr().out == (
        f"Installed: raft-ft {__version__}\n"
        f"Latest:    raft-ft {__version__} (PyPI)\n"
        "Status:    up to date\n"
    )


@pytest.mark.parametrize(
    ("installed", "latest", "status"),
    [
        ("3.1.0", "3.2.0", "update available"),
        ("3.2.0", "3.1.0", "installed version is newer than PyPI"),
    ],
)
def test_version_status_comparison(monkeypatch, capsys, installed, latest, status):
    monkeypatch.setattr(versioning, "__version__", installed)
    monkeypatch.setattr(versioning, "latest_pypi_version", lambda: latest)

    assert versioning.report_versions() == 0

    assert f"Status:    {status}\n" in capsys.readouterr().out


def test_version_action_reports_an_unavailable_latest_version(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["raft", "version"])
    failure = versioning.VersionCheckError("could not check PyPI: offline")

    with patch.object(versioning, "latest_pypi_version", side_effect=failure), \
         pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().out == (
        f"Installed: raft-ft {__version__}\n"
        "Latest:    unavailable (could not check PyPI: offline)\n"
    )


def test_invalid_installed_version_keeps_the_successful_pypi_result(monkeypatch, capsys):
    monkeypatch.setattr(versioning, "__version__", "not-a-version")
    monkeypatch.setattr(versioning, "latest_pypi_version", lambda: "3.1.0")

    assert versioning.report_versions() == 1

    assert capsys.readouterr().out.startswith(
        "Installed: raft-ft not-a-version\nLatest:    raft-ft 3.1.0 (PyPI)\n"
        "Status:    unable to compare versions ("
    )


def test_latest_version_uses_pypi_project_metadata():
    response = Mock()
    response.json.return_value = {"info": {"version": "3.2.0"}}

    with patch("raft.versioning.requests.get", return_value=response) as get:
        assert versioning.latest_pypi_version() == "3.2.0"

    response.raise_for_status.assert_called_once_with()
    assert get.call_args.args == (versioning.PYPI_JSON_URL,)
    assert get.call_args.kwargs["timeout"] == 10


@pytest.mark.parametrize(
    "failure",
    [
        requests.RequestException("offline"),
        None,
    ],
)
def test_latest_version_failures_are_clear(failure):
    response = Mock()
    response.json.return_value = {"no": "version"}
    side_effect = failure if failure is not None else None

    with patch("raft.versioning.requests.get", return_value=response, side_effect=side_effect):
        with pytest.raises(versioning.VersionCheckError, match="could not check PyPI"):
            versioning.latest_pypi_version()


def test_version_rejects_a_positional_argument(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["raft", "version", "extra"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert "unrecognized arguments: extra" in capsys.readouterr().err


def test_help_explains_the_two_version_forms(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["raft", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--version" in help_text
    assert "Print the installed raft-ft version and exit." in help_text
    assert "- version: Print the installed raft-ft version and compare it with PyPI." in help_text
