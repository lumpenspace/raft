"""CLI version reporting."""

import pytest

from raft import __version__, cli


@pytest.mark.parametrize("argv", [["raft", "version"], ["raft", "--version"]])
def test_version_works_outside_a_project(tmp_path, monkeypatch, capsys, argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", argv)

    if "version" not in argv:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    else:
        cli.main()

    assert capsys.readouterr().out == f"raft-ft {__version__}\n"


def test_version_rejects_a_positional_argument(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["raft", "version", "extra"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert "unrecognized arguments: extra" in capsys.readouterr().err


def test_help_advertises_both_version_forms(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["raft", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--version" in help_text
    assert "Print the installed raft-ft version and exit." in help_text
    assert "- version: Print the installed raft-ft version." in help_text
