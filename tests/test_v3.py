"""
Tests for the 2.3 five-phase interactive mode: document sources (feeds,
pages, PDFs), dataset state, and phase suggestion.

Everything here is offline -- no network, OpenAI, ariadne or opbdh
calls; HTTP is stubbed at sources._http_get.

    python -m unittest discover tests
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from raft import sources, state
from raft.flows import suggest_phase
from raft.sources import (
    append_corpus_records,
    date_num,
    extract_page,
    iso_date,
    parse_feed,
)

# Long enough to clear FULL_PAGE_THRESHOLD, so the entry counts as full.
FULL_BODY = "Full &lt;b&gt;body&lt;/b&gt; of the first post. " + "More prose. " * 60

RSS_FEED = f"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Unstable Ontology</title>
    <item>
      <title>First post</title>
      <link>https://example.com/first</link>
      <pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
      <description>teaser</description>
      <content:encoded>&lt;p&gt;{FULL_BODY}&lt;/p&gt;</content:encoded>
    </item>
    <item>
      <title>Second post</title>
      <link>https://example.com/second</link>
      <pubDate>Tue, 02 Jan 2024 12:00:00 +0000</pubDate>
      <description>only a teaser here</description>
    </item>
  </channel>
</rss>
"""

ATOM_FEED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom blog</title>
  <entry>
    <title>Atom entry</title>
    <link rel="alternate" href="https://example.com/atom-entry"/>
    <published>2024-03-04T05:06:07Z</published>
    <content type="html">&lt;p&gt;Atom content.&lt;/p&gt;</content>
  </entry>
</feed>
"""

PAGE_HTML = """<html><head>
  <title>Fallback title</title>
  <meta property="og:title" content="A long article"/>
  <meta property="article:published_time" content="2024-01-02T03:04:05Z"/>
</head><body>
  <nav>chrome to ignore</nav>
  <article><p>The article body.</p><p>Second paragraph.</p></article>
</body></html>
"""


class TempCwdTestCase(unittest.TestCase):
    """Run each test in a throwaway working directory (modules write to data/)."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp)
        os.makedirs("data", exist_ok=True)

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def read_jsonl(self, path: str) -> list:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]


class TestDates(unittest.TestCase):
    def test_iso_date_formats(self):
        self.assertEqual(iso_date("2024-01-02T03:04:05Z"), "2024-01-02")
        self.assertEqual(iso_date("Mon, 01 Jan 2024 12:00:00 +0000"), "2024-01-01")
        self.assertEqual(iso_date("not a date"), "")
        self.assertEqual(iso_date(None), "")

    def test_date_num(self):
        self.assertEqual(date_num("2024-01-02"), 20240102)
        self.assertEqual(date_num("Mon, 01 Jan 2024 12:00:00 +0000"), 20240101)
        self.assertEqual(date_num("unknown"), 0)
        self.assertEqual(date_num(""), 0)


class TestFeedParsing(unittest.TestCase):
    def test_rss_with_content_encoded(self):
        entries = parse_feed(RSS_FEED)
        self.assertEqual(len(entries), 2)
        first = entries[0]
        self.assertEqual(first["title"], "First post")
        self.assertEqual(first["link"], "https://example.com/first")
        self.assertEqual(first["date"], "2024-01-01")
        self.assertIn("Full", first["html"])
        # content:encoded is preferred over the teaser description
        self.assertNotIn("teaser", first["html"])
        self.assertEqual(entries[1]["html"], "only a teaser here")

    def test_atom(self):
        entries = parse_feed(ATOM_FEED)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["link"], "https://example.com/atom-entry")
        self.assertEqual(entries[0]["date"], "2024-03-04")
        self.assertIn("Atom content", entries[0]["html"])


class TestExtractPage(unittest.TestCase):
    def test_article_og_title_and_date(self):
        record = extract_page(PAGE_HTML, "https://example.com/a")
        self.assertEqual(record["title"], "A long article")
        self.assertEqual(record["date"], "2024-01-02")
        self.assertEqual(record["link"], "https://example.com/a")
        self.assertIn("The article body.", record["content"])
        self.assertNotIn("chrome to ignore", record["content"])

    def test_unknown_date_stays_unknown(self):
        # An undated page must not be stamped with today: under the
        # earlier-writings filter that would hide it from every
        # interview that predates the import.
        record = extract_page("<html><body><p>x</p></body></html>", "https://e/x")
        self.assertEqual(record["date"], "")
        self.assertEqual(date_num(record["date"]), 0)


class TestMarkupSafety(unittest.TestCase):
    def test_hx_escapes_bracketed_text(self):
        from raft import hx

        self.assertEqual(hx.esc("[/] and list[int]"), r"\[/] and list\[int]")


class TestMenuNavigation(unittest.TestCase):
    def test_menu_step_keys(self):
        from raft.hx import _menu_step

        self.assertEqual(_menu_step("down", 0, 3), (1, False))
        self.assertEqual(_menu_step("j", 1, 3), (2, False))
        self.assertEqual(_menu_step("j", 2, 3), (0, False))  # wraps
        self.assertEqual(_menu_step("up", 0, 3), (2, False))  # wraps
        self.assertEqual(_menu_step("k", 2, 3), (1, False))
        self.assertEqual(_menu_step("3", 0, 3), (2, False))  # digit jumps
        self.assertEqual(_menu_step("9", 1, 3), (1, False))  # out of range
        self.assertEqual(_menu_step("\r", 1, 3), (1, True))
        self.assertEqual(_menu_step("\n", 2, 3), (2, True))
        self.assertEqual(_menu_step("x", 1, 3), (1, False))  # unknown key
        self.assertEqual(_menu_step("", 1, 3), (1, False))  # bare ESC

    def test_choose_falls_back_without_a_tty(self):
        # Piped runs must keep working: no tty means the numbered prompt.
        import io
        import sys

        from raft.hx import _choose_with_keys

        with mock.patch.object(sys, "stdin", io.StringIO()):
            self.assertIsNone(_choose_with_keys("Pick", ["a", "b"], 0))


class TestCorpusAppend(TempCwdTestCase):
    def test_dedupes_by_link_across_runs(self):
        records = [
            {"title": "a", "link": "https://x/a", "date": "2024-01-01", "content": "A"},
            {"title": "b", "link": "https://x/b", "date": "2024-01-02", "content": "B"},
        ]
        self.assertEqual(append_corpus_records("d", records), 2)
        again = records + [
            {"title": "c", "link": "https://x/c", "date": "2024-01-03", "content": "C"}
        ]
        self.assertEqual(append_corpus_records("d", again), 1)
        self.assertEqual(len(self.read_jsonl("data/d.jsonl")), 3)

    def test_fetch_feed_fetches_teaser_pages(self):
        pages = {
            "https://blog.example/feed": RSS_FEED,
            "https://example.com/second": PAGE_HTML,
        }
        with mock.patch.object(sources, "_http_get", side_effect=pages.__getitem__):
            with mock.patch.object(sources.time, "sleep"):
                added = sources.fetch_feed("d", "https://blog.example/feed")
        self.assertEqual(added, 2)
        docs = self.read_jsonl("data/d.jsonl")
        # the first entry had full content:encoded; the second was a
        # teaser, so its linked page was fetched and extracted instead
        self.assertIn("Full", docs[0]["content"])
        self.assertIn("The article body.", docs[1]["content"])

    def test_import_pdf_missing_dependency_hint(self):
        try:
            import pypdf  # noqa: F401

            self.skipTest("pypdf installed; the hint path is not reachable")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError):
            sources.import_pdf("d", "whatever.pdf")


class TestState(TempCwdTestCase):
    def test_meta_roundtrip_and_legacy_model_id(self):
        with open("data/d_meta.json", "w") as f:
            json.dump({"model_id": "ft:gpt-3.5:old"}, f)
        self.assertEqual(state.finetuned_model("d"), "ft:gpt-3.5:old")
        state.record_finetuned_model("d", "ft:gpt-4o-mini:new", "openai")
        self.assertEqual(state.finetuned_model("d"), "ft:gpt-4o-mini:new")
        models = state.load_meta("d")["finetuned_models"]
        self.assertEqual([m["model"] for m in models], ["ft:gpt-3.5:old", "ft:gpt-4o-mini:new"])

    def test_model_backend_recorded_and_used_by_serve(self):
        from raft.serve import is_local_adapter

        state.record_finetuned_model("d", "runpod_results/r1/results/adapter", "hf")
        state.record_finetuned_model("d", "ft:gpt-4o-mini:x", "openai")
        self.assertEqual(state.model_backend("d"), "openai")
        self.assertEqual(
            state.model_backend("d", "runpod_results/r1/results/adapter"), "hf"
        )
        # recorded backend is authoritative, no path heuristics needed
        self.assertTrue(is_local_adapter("d", "runpod_results/r1/results/adapter"))
        self.assertFalse(is_local_adapter("d", "ft:gpt-4o-mini:x"))
        # hand-typed models fall back to the heuristic
        self.assertFalse(is_local_adapter("d", "ft:gpt-4o:org/suffix:abc"))

    def test_pending_job_survives_meta_roundtrip(self):
        state.update_meta("d", pending_oai_job={"id": "ftjob-1", "model": "m"})
        self.assertEqual(state.load_meta("d")["pending_oai_job"]["id"], "ftjob-1")
        state.update_meta("d", pending_oai_job=None)
        self.assertFalse(state.load_meta("d").get("pending_oai_job"))

    def test_test_questions_dedupe(self):
        state.add_test_question("d", "who are you?")
        state.add_test_question("d", "who are you?")
        state.add_test_question("d", "why llms?")
        self.assertEqual(state.test_questions("d"), ["who are you?", "why llms?"])

    def test_dataset_status_counts(self):
        with open("data/d.jsonl", "w") as f:
            f.write('{"title": "t", "content": "c"}\n' * 3)
        with open("data/d_transcript_1.json", "w") as f:
            f.write("{}")
        with open("data/d_transcript_benchmark.json", "w") as f:
            f.write("{}")
        status = state.dataset_status("d")
        self.assertEqual(status["corpus_docs"], 3)
        self.assertEqual(status["transcripts"], 1)  # benchmark not counted
        self.assertTrue(status["benchmark"])
        self.assertFalse(status["embedded"])
        self.assertEqual(status["model"], "")

    def test_list_datasets_skips_artifacts(self):
        for filename in (
            "d.jsonl",
            "d_chunked.jsonl",
            "d_finetune_openai.jsonl",
            "other_meta.json",
        ):
            with open(f"data/{filename}", "w") as f:
                f.write("{}\n")
        self.assertEqual(state.list_datasets(), ["d", "other"])


class TestSuggestPhase(unittest.TestCase):
    def status(self, **overrides):
        base = {
            "corpus_docs": 0,
            "chunks": 0,
            "embedded": False,
            "transcripts": 0,
            "finetune_file": False,
            "openai_file": False,
            "model": "",
            "benchmark": False,
            "questions": 0,
            "evaluated": False,
        }
        base.update(overrides)
        return base

    def test_progression(self):
        self.assertEqual(suggest_phase(self.status()), 0)
        self.assertEqual(suggest_phase(self.status(corpus_docs=2)), 1)
        self.assertEqual(
            suggest_phase(self.status(corpus_docs=2, openai_file=True)), 2
        )
        self.assertEqual(
            suggest_phase(
                self.status(corpus_docs=2, openai_file=True, model="ft:x")
            ),
            3,
        )
        self.assertEqual(
            suggest_phase(
                self.status(
                    corpus_docs=2, openai_file=True, model="ft:x", evaluated=True
                )
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
