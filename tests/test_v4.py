"""
2.8: dated conversations, recall and reasoning in thinking blocks,
conversation memories. Offline -- the LLM and embeddings are stubbed.
"""
import json
from unittest.mock import patch

import pytest

from raft import cli, embeddings_helpers, generate_finetune, memories, oai_finetune, state
from raft.convo_structurer import write_transcript
from raft.memories import MemoryManager, MetaDataKeyEnum
from raft.oai_finetune import oaify_example, think_block
from raft.prompt_manager import PromptManager
from raft.project import initialize_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return initialize_project(tmp_path / "persona")[0]


def test_transcript_carries_its_setting(project):
    path = write_transcript(project, {"q": "Pat", "a": "Sam"}, "2024-01-01", "u", [["Why?", "Because."]],
                            context="a reply thread on X")
    assert json.loads(open(path).read())["context"] == "a reply thread on X"
    path = write_transcript(project, {"q": "Pat", "a": "Sam"}, "2024-01-01", "u", [["Why?", "Because."]])
    assert "context" not in json.loads(open(path).read())


def test_system_prompt_states_persona_date_and_setting():
    plain = PromptManager().get_interview_system_message("Pat", "Sam", "2024-01-01")
    assert plain["content"].startswith("You are Sam. It is 2024-01-01. This is an interview; Pat is talking to you.")
    assert "recalled for you" in plain["content"]
    forum = PromptManager().get_interview_system_message(
        "Pat", "Sam", "2024-01-01", context='a LessWrong comment thread under the post "X"', thinking=True)
    assert 'This is a LessWrong comment thread under the post "X"; Pat is talking to you.' in forum["content"]
    assert "Before you reply, think" in forum["content"]
    assert "retrieve_memories" not in forum["content"]


def test_thinking_format_puts_recall_and_reasoning_in_the_reply():
    example = {"question": "Why?", "answer": "Because.", "similar_memories": "from 2023: I said so.\n",
               "reasoning": "That still holds."}
    messages, _ = oaify_example(example, {"q": "Pat", "a": "Sam"}, thinking=True)
    assert [m["role"] for m in messages] == ["user", "assistant"]  # no memories system note
    assert messages[1]["content"] == (
        "<think>\nRecalling what I have written before:\nfrom 2023: I said so.\n\nThat still holds.\n</think>\n\nBecause."
    )
    bare = think_block({"question": "Why?", "answer": "Because."})
    assert bare == "<think>\nNothing I have written before bears on this directly.\n</think>\n\n"
    plain, _ = oaify_example(example, {"q": "Pat", "a": "Sam"})
    assert [m["role"] for m in plain] == ["user", "system", "assistant"] and plain[2]["content"] == "Because."


def test_generation_writes_context_and_reasoning(project):
    write_transcript(project, {"q": "Pat", "a": "Sam"}, "2024-05-01", "u", [["Why?", "Because."], ["And?", "So."]],
                     context="a reply thread on X")
    with patch.object(MemoryManager, "__init__", lambda self, *a, **k: None), \
         patch.object(MemoryManager, "get_similar_and_summarize", return_value="from 2023: I said so.") as recall, \
         patch.object(MemoryManager, "reasoning_trace", return_value="Hence.") as trace:
        generate_finetune.generate_finetune(project, thinking=True)
    assert recall.call_args_list[0].kwargs["store"] is True
    assert trace.call_args_list[1].args == ("And?", "So.", "from 2023: I said so.", "Because.")
    rows = json.loads(project.finetune_path.read_text())
    assert rows[0]["metadata"]["context"] == "a reply thread on X" and rows[0]["metadata"]["date"] == "2024-05-01"
    assert rows[1]["example"] == {"question": "Why?", "answer": "Because.",
                                  "similar_memories": "from 2023: I said so.", "reasoning": "Hence."}
    oai_finetune.create_openai_finetune_file(project, thinking=True)
    examples = [json.loads(line) for line in project.finetune_openai_path.read_text().splitlines()]
    assert examples[0]["messages"][0]["content"].startswith("You are Sam. It is 2024-05-01. This is a reply thread on X")
    assert examples[0]["messages"][-1]["content"].startswith("<think>\nRecalling what I have written before:")
    assert examples[0]["messages"][-1]["content"].endswith("Hence.\n</think>\n\nBecause.")
    # the second exchange carries the first as prior turns, within one conversation
    assert [m["role"] for m in examples[1]["messages"]] == ["system", "user", "assistant", "user", "assistant"]


def test_benchmark_is_never_remembered(project):
    write_transcript(project, {"q": "Pat", "a": "Sam"}, "2024-05-01", "u", [["Why?", "Because."]], index="benchmark")
    with patch.object(MemoryManager, "__init__", lambda self, *a, **k: None), \
         patch.object(MemoryManager, "get_similar_and_summarize", return_value="") as recall:
        generate_finetune.generate_benchmark(project)
    assert recall.call_args.kwargs["store"] is False


def test_exchanges_are_remembered_dated_and_only_after_the_query(project):
    calls = []
    manager = MemoryManager.__new__(MemoryManager)
    manager.dataset = project
    manager.name = "sam"
    manager.metadata = {MetaDataKeyEnum.DATE: "2024-05-01", MetaDataKeyEnum.URL: "u",
                        MetaDataKeyEnum.PARTICIPANTS: {"q": "Pat", "a": "Sam"}}
    manager._dated = True

    class Collection:
        def query(self, **kwargs):
            calls.append(("query", kwargs.get("where")))
            return {"metadatas": [[]], "documents": [[]]}

    manager.collection = Collection()
    with patch("raft.memories.get_embedding", return_value=[0.1, 0.2]), \
         patch("raft.memories.store_exchange_embedding", side_effect=lambda *a: calls.append(("store", a[0]))) as store:
        manager.get_similar_extracts(["Why?", "Because."], store=True)
    assert calls == [("query", {"date_num": {"$lt": 20240501}}), ("store", {"question": "Why?", "answer": "Because."})]
    store.assert_called_once()

    stored = {}

    class Chroma:
        def get_or_create_collection(self, name):
            return self

        def upsert(self, **kwargs):
            stored.update(kwargs)

    with patch("raft.embeddings_helpers.PersistentClient", return_value=Chroma()):
        embeddings_helpers.store_exchange_embedding(
            {"question": "Why?", "answer": "Because."}, project,
            {"date": "2024-05-01", "url": "u", "participants": {"q": "Pat", "a": "Sam"}}, [0.1])
    assert stored["documents"] == "Pat: Why?\nSam: Because."
    assert stored["metadatas"]["date_num"] == 20240501 and stored["metadatas"]["kind"] == "exchange"
    assert stored["metadatas"]["participants"] == "Pat, Sam"


def test_ask_question_splits_thinking_from_the_reply(project):
    with patch("raft.memories.OpenAI") as client:
        message = client.return_value.chat.completions.create.return_value.choices[0].message
        message.content = "<think>\nRecalling...\n</think>\n\nBecause."
        message.reasoning_content = None
        manager = MemoryManager(project, {})
        with patch("raft.memories.hx.say") as say:
            assert manager.ask_question("Why?", model="m", thinking=True) == "Because."
        assert "Recalling" in say.call_args.args[0]
    system = client.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert system.startswith("You are persona. It is ") and "Before you reply, think" in system


def test_thinking_flag_sticks_to_the_dataset(project, monkeypatch):
    monkeypatch.chdir(project.root)
    monkeypatch.setattr("sys.argv", ["raft", "ft:gen", "--generic", "--thinking"])
    with patch.object(generate_finetune, "generate_finetune") as generate:
        cli.main()
    assert generate.call_args.kwargs["thinking"] is True
    assert state.load_meta(project)["thinking"] is True
    monkeypatch.setattr("sys.argv", ["raft", "ft:gen", "--oai"])
    with patch.object(oai_finetune, "create_openai_finetune_file") as convert:
        cli.main()
    assert convert.call_args.kwargs["thinking"] is True


def test_pace_defaults_to_no_waiting():
    assert memories.PACE == 0


DOC = "Bigger neural nets ought to have higher inference latency in general, regardless of pipelining. Other stuff."


def test_recall_must_cite_the_material():
    from raft.memories import grounded_recall

    assert grounded_recall("skip", DOC) == ""
    assert grounded_recall("Skip.\nSKIP I've argued that x.", DOC) == ""
    assert grounded_recall("I've argued that x.", DOC) == ""  # unstructured: not trusted
    good = "SOURCE: Bigger neural nets ought to have higher inference latency in general, regardless of pipelining.\nRECALL: I've argued that bigger nets mean higher latency."
    assert grounded_recall(good, DOC) == "I've argued that bigger nets mean higher latency."
    edited = 'SOURCE: "bigger neural nets ought to have higher inference latency"\nRECALL: I said latency grows.'
    assert grounded_recall(edited, DOC) == "I said latency grows."
    invented = "SOURCE: LLMs sample high-probability responses from their training data.\nRECALL: I've argued that LLMs mode-collapse."
    assert grounded_recall(invented, DOC) == ""
    two = ("SOURCE: Bigger neural nets ought to have higher inference latency in general.\nRECALL: Latency grows with size.\n"
           "SOURCE: LLMs sample high-probability responses.\nRECALL: I said LLMs mode-collapse.\n"
           "SOURCE: regardless of pipelining. Other stuff.\nRECALL: Pipelining does not save you.")
    assert grounded_recall(two, DOC) == "Latency grows with size. Pipelining does not save you."


def test_summaries_go_through_the_grounding_check(project):
    manager = MemoryManager.__new__(MemoryManager)
    manager.name = "sam"
    manager.prompt_manager = PromptManager()
    reply = "SOURCE: Other stuff.\nRECALL: I've said other stuff."
    with patch.object(PromptManager, "summarize_memory", return_value=reply) as summarize:
        got = manager.summarize_memory({"date": "2024-01-01", "title": "Latency", "document": DOC}, "q", "")
    assert got["memory"] == ""  # a two-word citation is not enough anchoring
    assert summarize.call_args.kwargs["title"] == "Latency" and summarize.call_args.kwargs["date"] == "2024-01-01"


def test_serve_chats_with_a_local_model_behind_an_endpoint(project, monkeypatch):
    from raft import serve

    state.record_finetuned_model(project, "/models/adapter", "hf")
    with patch("raft.serve.ask", side_effect=[""]), patch("raft.serve.MemoryManager") as manager, \
         patch("raft.serve.hx.say") as say:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        serve.run_serve(project, standalone=False)
        assert "mlx_lm.server" in say.call_args.args[0]
        manager.assert_not_called()
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
        serve.run_serve(project, standalone=False)
        manager.assert_called_once()


def test_prompts_trim_the_previous_reply_and_pick_the_reasoning_model(monkeypatch):
    from raft import prompt_manager as pm

    long_reply = "word " * 500
    manager = PromptManager()
    with patch.object(PromptManager, "client") as client:
        client.chat.completions.create.return_value.choices[0].message.content = "ok"
        manager.summarize_memory("m", "q", long_reply, author="Sam")
        sent = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "[...]" in sent and len(sent) < 1200
        manager.reasoning_trace("q", "a", "", long_reply, author="Sam")
        assert client.chat.completions.create.call_args.kwargs["model"] == pm.REASONING_MODEL
    assert pm.REASONING_MODEL == pm.SUMMARY_MODEL  # defaults to the summariser's model


def test_a_failed_summary_is_skipped_not_fatal():
    manager = MemoryManager.__new__(MemoryManager)
    manager.name = "sam"
    manager.prompt_manager = PromptManager()
    with patch.object(PromptManager, "summarize_memory", side_effect=RuntimeError("500")), patch("raft.memories.hx.warn"):
        assert manager.summarize_memory({"date": "2024-01-01", "document": "d"}, "q", "") == {"date": "2024-01-01", "memory": ""}


def test_each_role_can_have_its_own_endpoint(monkeypatch):
    from raft import convo_structurer
    from raft import prompt_manager as pm

    monkeypatch.setenv("RAFT_LLM_BASE_URL", "http://helper:1/v1")
    monkeypatch.setenv("RAFT_LLM_API_KEY", "h")
    monkeypatch.setenv("RAFT_EMBEDDING_BASE_URL", "http://embed:2/v1")
    monkeypatch.setenv("RAFT_EMBEDDING_API_KEY", "e")
    monkeypatch.setenv("OPENAI_API_KEY", "persona")
    embeddings_helpers._client = None
    assert str(embeddings_helpers._get_client().base_url) == "http://embed:2/v1/"
    assert str(PromptManager().client.base_url) == "http://helper:1/v1/"
    with patch("raft.prompt_manager.OpenAI") as client:
        convo_structurer.helper_client()
    assert client.call_args.kwargs == {"base_url": "http://helper:1/v1", "api_key": "h"}
    assert pm.REASONING_MODEL  # the persona's own client stays OPENAI_BASE_URL / OPENAI_API_KEY
    embeddings_helpers._client = None


def test_text_about_special_tokens_chunks_and_counts(project):
    from raft import files_helper

    project.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    project.corpus_path.write_text(json.dumps({"title": "t", "link": "l", "date": "2024-01-01",
                                               "content": "GPT stops at <|endoftext|> and <|im_end|>."}) + "\n")
    files_helper.chunker(project)
    [(meta, chunk)] = [json.loads(line) for line in project.chunks_path.read_text().splitlines()]
    assert "<|endoftext|>" in chunk and meta["title"] == "t"
    assert oai_finetune.count_tokens({"content": "<|endoftext|>"}) > 0
