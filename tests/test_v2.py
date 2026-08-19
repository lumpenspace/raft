"""
Tests for the 2.0 dataset-building pipeline: conversation structuring,
tweet-mode import, and huggingface run preparation.

Everything here is offline -- no OpenAI, ariadne or opbdh calls.

    poetry run python -m unittest discover tests
"""

import ast
import json
import os
import shutil
import tempfile
import unittest

from raft.convo_structurer import (
    find_message_arrays,
    import_conversation_file,
    import_text_source_file,
    is_transcript,
    messages_to_exchanges,
    next_transcript_index,
    parse_json_or_jsonl,
)
from raft.hf_finetune import (
    coerce_flag_value,
    is_openai_finetunable,
    parse_opbdh_flags,
    prepare_run_dir,
)
from raft.tweet_mode import import_documents, iso_date, normalize_messages


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

    def write(self, filename: str, payload: object) -> str:
        with open(filename, "w") as f:
            if isinstance(payload, str):
                f.write(payload)
            else:
                json.dump(payload, f)
        return filename

    def read_json(self, path: str) -> object:
        with open(path) as f:
            return json.load(f)

    def read_jsonl(self, path: str) -> list:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def read_text(self, path: str) -> str:
        with open(path) as f:
            return f.read()


class TestConversationStructuring(TempCwdTestCase):
    def test_transcript_passthrough(self):
        source = {
            "participants": {"q": "Bob", "a": "gary"},
            "date": "2020-01-01",
            "url": "http://example.com",
            "exchanges": [["x", "y"]],
        }
        self.write("t.json", source)
        result = self.read_json(import_conversation_file("d", "t.json", "gary"))
        self.assertTrue(is_transcript(result))
        self.assertEqual(result["participants"]["q"], "Bob")
        self.assertEqual(result["date"], "2020-01-01")

    def test_message_array_merges_consecutive_turns(self):
        self.write(
            "c.json",
            {
                "messages": [
                    {"role": "user", "name": "alice", "content": "Why parrots?"},
                    {"role": "assistant", "name": "gary", "content": "Stochastic."},
                    {"role": "user", "name": "alice", "content": "Really?"},
                    {"role": "user", "name": "alice", "content": "Truly?"},
                    {"role": "assistant", "name": "gary", "content": "Yes."},
                ]
            },
        )
        result = self.read_json(import_conversation_file("d", "c.json", "gary"))
        self.assertEqual(
            result["exchanges"],
            [["Why parrots?", "Stochastic."], ["Really?\nTruly?", "Yes."]],
        )
        self.assertEqual(result["participants"], {"q": "alice", "a": "gary"})

    def test_ariadne_conversations_wrapper(self):
        self.write(
            "a.json",
            {
                "format": "messages",
                "conversations": [
                    {
                        "target_id": "1",
                        "messages": [
                            {"role": "user", "name": "rando", "content": "take?"},
                            {"role": "assistant", "name": "gary", "content": "one"},
                        ],
                    },
                    {
                        "target_id": "2",
                        "messages": [
                            {"role": "user", "name": "rando", "content": "another?"},
                            {"role": "assistant", "name": "gary", "content": "two"},
                        ],
                    },
                ],
            },
        )
        result = self.read_json(import_conversation_file("d", "a.json", "gary"))
        self.assertEqual(
            result["exchanges"], [["take?", "one"], ["another?", "two"]]
        )

    def test_jsonl_lines_of_messages(self):
        with open("l.jsonl", "w") as f:
            for i in range(2):
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": f"q{i}"},
                                {"role": "assistant", "content": f"a{i}"},
                            ]
                        }
                    )
                    + "\n"
                )
        result = self.read_json(import_conversation_file("d", "l.jsonl", "gary"))
        self.assertEqual(result["exchanges"], [["q0", "a0"], ["q1", "a1"]])

    def test_target_name_identifies_answers_without_assistant_role(self):
        messages = [
            {"role": "user", "name": "alice", "content": "q"},
            {"role": "user", "name": "gary", "content": "a"},
        ]
        exchanges, questioner = messages_to_exchanges(messages, "gary")
        self.assertEqual(exchanges, [["q", "a"]])
        self.assertEqual(questioner, "alice")

    def test_raw_text_is_not_mistaken_for_structured_input(self):
        self.assertIsNone(parse_json_or_jsonl("just some prose"))
        self.assertEqual(find_message_arrays(None), [])

    def test_transcripts_do_not_overwrite_each_other(self):
        self.write("a.json", [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "a"}])
        first = import_conversation_file("d", "a.json", "gary")
        second = import_conversation_file("d", "a.json", "gary")
        self.assertNotEqual(first, second)
        self.assertEqual(next_transcript_index("d"), 3)


class TestTextSources(TempCwdTestCase):
    def test_raw_text_becomes_one_grounding_document(self):
        self.write("essay.txt", "a long essay about parrots")
        count = import_text_source_file("d", "essay.txt")
        self.assertEqual(count, 1)
        record = self.read_jsonl("data/d.jsonl")[0]
        self.assertEqual(record["content"], "a long essay about parrots")
        self.assertEqual(record["title"], "essay")

    def test_structured_jsonl_passes_through_and_appends(self):
        with open("posts.jsonl", "w") as f:
            f.write(json.dumps({"title": "t1", "link": "u1",
                                "date": "2024-01-01", "content": "c1"}) + "\n")
            f.write(json.dumps({"title": "t2", "link": "u2",
                                "date": "2024-01-02", "content": "c2"}) + "\n")
        self.assertEqual(import_text_source_file("d", "posts.jsonl"), 2)
        self.assertEqual(import_text_source_file("d", "posts.jsonl"), 2)
        records = self.read_jsonl("data/d.jsonl")
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["title"], "t1")


class TestTweetMode(TempCwdTestCase):
    def documents(self, n=3):
        """Documents shaped exactly as ariadne's `raft.documents.v1` renderer
        emits them: message bodies under `text`, and the thread root tagged
        "assistant" with every later reply tagged "participant"."""
        return [
            {
                "format": "raft.documents.v1",
                "id": f"ariadne:{i}",
                "source": "ariadne",
                "kind": "tweet_conversation",
                "text": f"@rando: hot take {i}?\n\n@gary: reply {i}",
                "metadata": {
                    "target_id": str(i),
                    "target_author": "@gary",
                    "target_created_at": "2024-03-01T10:00:00Z",
                    "target_url": f"https://x.com/gary/status/{i}",
                    "tweet_ids": [str(i)],
                },
                "messages": [
                    {
                        "role": "assistant",
                        "tweet_id": f"{i}0",
                        "author": "@rando",
                        "username": "rando",
                        "text": f"hot take {i}?",
                    },
                    {
                        "role": "participant",
                        "tweet_id": str(i),
                        "author": "@gary",
                        "username": "gary",
                        "text": f"reply {i}",
                    },
                ],
            }
            for i in range(n)
        ]

    def test_normalize_maps_ariadne_message_shape(self):
        # Bodies come from `text`, names from `username`, and the target's
        # turns become the answers regardless of ariadne's positional roles.
        messages = normalize_messages(self.documents(1)[0]["messages"], "gary")
        self.assertEqual(
            [(m["role"], m["name"], m["content"]) for m in messages],
            [("user", "rando", "hot take 0?"), ("assistant", "gary", "reply 0")],
        )

    def test_normalize_falls_back_to_positional_roles_without_a_match(self):
        messages = normalize_messages(self.documents(1)[0]["messages"], "someone-else")
        self.assertEqual(
            [(m["role"], m["name"]) for m in messages],
            [("assistant", "rando"), ("user", "gary")],
        )

    def test_target_username_decides_answers_over_thread_position(self):
        # Ariadne tags the thread ROOT "assistant"; here the root is someone
        # else, so the target's reply must still come out as the answer.
        import_documents("tw", self.documents(1), "gary")
        transcript = self.read_json("data/tw_transcript_1.json")
        self.assertEqual(transcript["exchanges"], [["hot take 0?", "reply 0"]])

    def test_target_handle_matches_with_or_without_at(self):
        import_documents("tw", self.documents(1), "@gary")
        transcript = self.read_json("data/tw_transcript_1.json")
        self.assertEqual(transcript["exchanges"], [["hot take 0?", "reply 0"]])

    def test_archive_style_dates_are_normalized(self):
        # Archive exports use Twitter's legacy stamp, not ISO 8601.
        self.assertEqual(iso_date("Mon Jan 01 12:05:00 +0000 2024"), "2024-01-01")
        self.assertEqual(iso_date("2024-03-01T10:00:00Z"), "2024-03-01")
        self.assertEqual(iso_date("2024-03-01"), "2024-03-01")
        self.assertEqual(iso_date(""), "")
        self.assertEqual(iso_date(None), "")
        self.assertEqual(iso_date("not a date"), "")

    def test_legacy_dates_survive_the_import(self):
        documents = self.documents(1)
        documents[0]["metadata"]["target_created_at"] = "Mon Jan 01 12:05:00 +0000 2024"
        import_documents("tw", documents, "gary")
        self.assertEqual(self.read_jsonl("data/tw.jsonl")[0]["date"], "2024-01-01")

    def test_self_thread_keeps_grounding_but_writes_no_transcript(self):
        documents = self.documents(1)
        for message in documents[0]["messages"]:
            message["username"] = "gary"
        import_documents("tw", documents, "gary")
        self.assertEqual(len(self.read_jsonl("data/tw.jsonl")), 1)
        self.assertFalse(os.path.exists("data/tw_transcript_1.json"))

    def test_import_builds_corpus_and_transcripts(self):
        import_documents("tw", self.documents(3), "gary")
        corpus = self.read_jsonl("data/tw.jsonl")
        self.assertEqual(len(corpus), 3)
        self.assertEqual(corpus[0]["content"], "@rando: hot take 0?\n\n@gary: reply 0")
        self.assertEqual(corpus[0]["date"], "2024-03-01")
        transcript = self.read_json("data/tw_transcript_1.json")
        self.assertEqual(len(transcript["exchanges"]), 3)
        self.assertEqual(transcript["participants"]["a"], "gary")

    def test_duplicate_documents_are_dropped(self):
        docs = self.documents(2)
        import_documents("tw", docs + docs, "gary")
        corpus = self.read_jsonl("data/tw.jsonl")
        self.assertEqual(len(corpus), 2)


class TestOpbdhFlagParsing(TempCwdTestCase):
    def test_flags_become_typed_keyword_overrides(self):
        self.assertEqual(
            parse_opbdh_flags(["--vram-gb", "48", "--max-spend", "5"]),
            {"vram_gb": 48, "max_spend": 5},
        )

    def test_equals_form_and_bare_switches(self):
        self.assertEqual(
            parse_opbdh_flags(["--auto-network-volume", "--provider", "primeintellect"]),
            {"auto_network_volume": True, "provider": "primeintellect"},
        )

    def test_values_keep_their_natural_types(self):
        self.assertEqual(coerce_flag_value("48"), 48)
        self.assertEqual(coerce_flag_value("1.5"), 1.5)
        self.assertIs(coerce_flag_value("true"), True)
        self.assertEqual(coerce_flag_value("Qwen/Qwen2.5-7B"), "Qwen/Qwen2.5-7B")

    def test_empty_flags_produce_no_overrides(self):
        self.assertEqual(parse_opbdh_flags([]), {})


class TestHuggingfaceRun(TempCwdTestCase):
    def test_model_routing(self):
        self.assertTrue(is_openai_finetunable("gpt-4o-mini-2024-07-18"))
        self.assertTrue(is_openai_finetunable("gpt-3.5-turbo"))
        self.assertFalse(is_openai_finetunable("Qwen/Qwen2.5-7B-Instruct"))
        self.assertFalse(is_openai_finetunable("meta-llama/Llama-3.1-8B"))

    def test_run_dir_is_self_contained(self):
        with open("data/d_finetune_openai.jsonl", "w") as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "q", "name": "alice"},
                            {"role": "assistant", "content": "a", "name": "gary"},
                        ]
                    }
                )
                + "\n"
            )
        run_dir = prepare_run_dir("d")
        for filename in ("run.py", "requirements.txt", "train.jsonl"):
            self.assertTrue(os.path.exists(f"{run_dir}/{filename}"), filename)
        ast.parse(self.read_text(f"{run_dir}/run.py"))
        row = self.read_jsonl(f"{run_dir}/train.jsonl")[0]
        for message in row["messages"]:
            self.assertEqual(set(message), {"role", "content"})


if __name__ == "__main__":
    unittest.main()
