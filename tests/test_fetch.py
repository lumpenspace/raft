"""`raft fetch <source>`: routing, the 3.0 aliases, and the grounding-only sources. Offline."""
from unittest.mock import patch

import pytest

from raft import cli
from raft.project import initialize_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    project, _ = initialize_project(tmp_path / "p")
    monkeypatch.chdir(project.root)
    return project


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["raft", *argv])
    cli.main()


def test_fetch_needs_a_known_source(project, monkeypatch, capsys):
    for argv in (["fetch"], ["fetch", "reddit"]):
        monkeypatch.setattr("sys.argv", ["raft", *argv])
        with pytest.raises(SystemExit):
            cli.main()
        assert "fetch needs a source" in capsys.readouterr().err


def test_legacy_lesswrong_and_tweets_actions_still_route(project, monkeypatch, capsys):
    with patch("raft.lesswrong.import_lesswrong", return_value={}) as importer:
        run(monkeypatch, "lesswrong", "--user", "gwern", "--conversations", "3")
    assert importer.call_args.args[2] == "gwern" and importer.call_args.kwargs["max_conversations"] == 3
    assert "raft fetch lesswrong" in capsys.readouterr().err
    with patch("raft.tweet_mode.run_tweet_mode") as tweets:
        run(monkeypatch, "tweets", "--role", "corpus")
    assert tweets.call_args.kwargs["role"] == "corpus"


def test_substack_rss_url_and_pdf_take_their_flags(project, monkeypatch):
    with patch("raft.flows.add_substack") as substack:
        run(monkeypatch, "fetch", "substack", "--blog", "garymarcus")
    assert substack.call_args.args[1] == "garymarcus"

    with patch("raft.sources.fetch_feed", return_value=2) as feed:
        run(monkeypatch, "fetch", "rss", "--url", "https://blog.example", "--no-full-pages")
    assert feed.call_args.args[1] == "https://blog.example"
    assert feed.call_args.kwargs == {"fetch_pages": False, "since": "", "until": "", "limit": None}

    with patch("raft.sources.fetch_url", return_value=1) as page:
        run(monkeypatch, "fetch", "url", "--url", "https://blog.example/essay")
    assert page.call_args.args[1] == "https://blog.example/essay"

    with patch("raft.sources.import_pdf", return_value=1) as pdf:
        run(monkeypatch, "fetch", "pdf", "--file", "a.pdf", "--file", "b.pdf")
    assert [c.args[1] for c in pdf.call_args_list] == ["a.pdf", "b.pdf"]


def test_fetch_errors_are_usage_errors(project, monkeypatch, capsys):
    with patch("raft.sources.fetch_feed", side_effect=ValueError("neither a feed nor a page")):
        with pytest.raises(SystemExit):
            run(monkeypatch, "fetch", "rss", "--url", "https://nope")
    assert "neither a feed" in capsys.readouterr().err


def test_dataset_name_still_follows_the_source_outside_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("raft.sources.fetch_url", return_value=1) as page:
        run(monkeypatch, "fetch", "url", "legacy", "--url", "https://x/y")
    assert page.call_args.args[0].name == "legacy"


def test_shared_window_and_limit_reach_every_source(project, monkeypatch):
    with patch("raft.flows.add_substack") as substack:
        run(monkeypatch, "fetch", "substack", "--blog", "b", "--since", "2024-01-01", "--until", "2024-06-30", "--limit", "5")
    assert substack.call_args.kwargs == {"since": "2024-01-01", "until": "2024-06-30", "limit": 5}
    with patch("raft.sources.fetch_feed", return_value=0) as feed:
        run(monkeypatch, "fetch", "rss", "--url", "https://b", "--limit", "3")
    assert feed.call_args.kwargs["limit"] == 3
    with patch("raft.tweet_mode.run_tweet_mode") as tweets:
        run(monkeypatch, "fetch", "tweets", "--user", "@gary", "--since", "2024-01-01", "--limit", "10", "--network", "x")
    assert tweets.call_args.kwargs == {"role": "auto", "user": "@gary", "archive": "", "network": "x",
                                       "since": "2024-01-01", "until": "", "limit": 10}


def test_bad_dates_are_usage_errors(project, monkeypatch, capsys):
    for argv in (["--since", "yesterday"], ["--since", "2024-02-01", "--until", "2024-01-01"]):
        with pytest.raises(SystemExit):
            run(monkeypatch, "fetch", "rss", "--url", "https://b", *argv)
        assert "--since" in capsys.readouterr().err


def test_select_records_window_and_newest_first():
    from raft.sources import in_window, select_records

    assert in_window("2024-05-05", "2024-01-01", "2024-12-31")
    assert not in_window("", "2024-01-01", "") and in_window("", "", "")
    assert not in_window("Mon, 01 Jan 2024 00:00:00 +0000", "2024-01-02", "")
    records = [{"date": "2023-01-01"}, {"date": "2024-03-01"}, {"date": ""}, {"date": "2024-01-01"}]
    assert select_records(records, since="2024-01-01") == [{"date": "2024-03-01"}, {"date": "2024-01-01"}]
    assert select_records(records, limit=2) == [{"date": "2024-03-01"}, {"date": "2024-01-01"}]
    assert select_records(records) == records


def test_substack_listing_stops_at_since_and_limit(monkeypatch):
    from raft import substack_embeddings

    listing = [[{"canonical_url": f"https://b/{d}", "title": d, "post_date": f"{d}T00:00:00Z"}
                for d in ("2024-03-01", "2024-02-01")],
               [{"canonical_url": "https://b/2023", "title": "old", "post_date": "2023-01-01T00:00:00Z"}], []]
    calls = []
    monkeypatch.setattr(substack_embeddings, "fetch_json", lambda url, params: (calls.append(params), listing[params["offset"] // 12])[1])
    monkeypatch.setattr(substack_embeddings, "fetch_html", lambda link: '<div class="markup">body</div>')
    monkeypatch.setattr(substack_embeddings, "sleep", lambda s: None)
    posts = list(substack_embeddings.fetch_and_parse("https://b", since="2024-01-01"))
    assert [p["title"] for p in posts] == ["2024-03-01", "2024-02-01"]
    assert len(calls) == 2  # the page holding an older post ends the listing
    posts = list(substack_embeddings.fetch_and_parse("https://b", limit=1))
    assert [p["title"] for p in posts] == ["2024-03-01"]


class FakeResult:
    def __init__(self, documents):
        self._documents, self.warnings = documents, []

    def raft_documents(self):
        return self._documents

    def save_cache(self, path=None):
        pass


def thread(day, ident):
    return {"id": ident, "text": f"thread {ident}",
            "metadata": {"target_created_at": f"{day}T10:00:00Z", "target_url": f"https://x.com/gary/status/{ident}"},
            "messages": [{"username": "amy", "text": "why?"}, {"username": "gary", "text": "because"}]}


def test_tweets_with_user_asks_nothing_and_applies_the_window(project, monkeypatch):
    from raft import state, tweet_mode

    docs = [thread("2024-03-01", "1"), thread("2024-01-01", "2"), thread("2023-06-01", "3")]
    fake = type("ariadne", (), {})()
    fake.build = lambda **options: (setattr(fake, "options", options), FakeResult(docs))[1]
    monkeypatch.setattr(tweet_mode, "load_ariadne", lambda: fake)
    monkeypatch.setattr(tweet_mode, "ask", lambda *a, **k: pytest.fail("asked"))
    monkeypatch.setattr(tweet_mode, "choose", lambda *a, **k: pytest.fail("asked"))

    run(monkeypatch, "fetch", "tweets", "--user", "@gary", "--since", "2024-01-01", "--limit", "1")
    assert fake.options == {"allow_empty": True, "community_archive": True, "target_user": "gary", "since": "2024-01-01",
                            "cache": str(project.ariadne_cache_path)}
    assert state.dataset_status(project)["transcripts"] == 1
    assert state.load_meta(project)["target"] == "gary"

    # an archive export, everyone's tweets unless --user narrows it
    fake.options = None
    run(monkeypatch, "fetch", "tweets", "--archive", "/tmp/export.zip", "--until", "2023-12-31")
    assert fake.options["archive"] == "/tmp/export.zip" and fake.options["all_loaded"] is True
    assert state.dataset_status(project)["transcripts"] == 2  # + the 2023 thread


def test_tweets_bluesky_is_guessed_from_a_dotted_handle(project, monkeypatch):
    from raft import tweet_mode

    fake = type("ariadne", (), {})()
    fake.build = lambda **o: pytest.fail("X branch ran")
    fake.build_bluesky = lambda handle, since=None, allow_empty=True: (setattr(fake, "seen", (handle, since)), FakeResult([]))[1]
    monkeypatch.setattr(tweet_mode, "load_ariadne", lambda: fake)
    monkeypatch.setattr(tweet_mode, "ask", lambda *a, **k: pytest.fail("asked"))
    with pytest.raises(SystemExit):  # nothing reconstructed
        run(monkeypatch, "fetch", "tweets", "--user", "alice.bsky.social", "--since", "2024-01-01")
    assert fake.seen == ("alice.bsky.social", "2024-01-01")
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "fetch", "tweets", "--user", "gary", "--network", "both")
    assert exc.value.code == 2
