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


def test_thinking_recall_rides_in_the_single_system_message(project):
    with patch("raft.memories.OpenAI") as client:
        client.return_value.chat.completions.create.return_value.choices[0].message.content = "ok"
        manager = MemoryManager(project, {})
        with patch.object(MemoryManager, "get_similar_and_summarize", return_value="from 2023: I said so."):
            manager.ask_question("Why?", model="m", thinking=True)
            roles = [m["role"] for m in client.return_value.chat.completions.create.call_args.kwargs["messages"]]
            first = client.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
            assert roles == ["system", "user"] and "from 2023: I said so." in first
            manager.ask_question("Why?", model="m", thinking=False)
            roles = [m["role"] for m in client.return_value.chat.completions.create.call_args.kwargs["messages"]]
            assert roles == ["system", "system", "user"]


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


def _manager():
    manager = MemoryManager.__new__(MemoryManager)
    manager.name = "sam"
    manager.prompt_manager = PromptManager()
    MemoryManager.reset_trace_stats()
    return manager


def test_trace_is_judged_and_rewritten_with_the_objection():
    manager = _manager()
    verdicts = iter([(False, "it leans on the recollection the reply never uses"), (True, "reaches it")])
    with patch.object(PromptManager, "reasoning_trace", side_effect=["bad", "good"]) as write, \
         patch.object(PromptManager, "check_trace", side_effect=lambda *a, **k: next(verdicts)), \
         patch("raft.memories.hx.say"):
        assert manager.reasoning_trace("q", "a", "m", "") == "good"
    assert write.call_args_list[0].kwargs["objection"] == ""
    assert "never uses" in write.call_args_list[1].kwargs["objection"]
    assert write.call_args_list[1].kwargs["rejected"] == "bad"  # the retry sees what was rejected
    assert MemoryManager.trace_stats == {"passed": 0, "rewritten": 1, "dropped": 0, "unjudged": 0}


def test_trace_that_never_leads_to_the_reply_is_dropped():
    manager = _manager()
    with patch.object(PromptManager, "reasoning_trace", return_value="bad") as write, \
         patch.object(PromptManager, "check_trace", return_value=(False, "drifts")), \
         patch("raft.memories.hx.say"), patch("raft.memories.hx.warn"):
        assert manager.reasoning_trace("q", "a", "", "") == ""
    assert write.call_count == MemoryManager.TRACE_REWRITES + 1
    assert MemoryManager.trace_stats["dropped"] == 1


def test_existing_trace_is_kept_when_it_passes():
    manager = _manager()
    with patch.object(PromptManager, "reasoning_trace") as write, \
         patch.object(PromptManager, "check_trace", return_value=(True, "fine")):
        assert manager.reasoning_trace("q", "a", "", "", existing="old") == "old"
    write.assert_not_called()


def test_check_trace_parses_the_checklist():
    from raft.prompt_manager import NO_VERDICT, parse_verdict

    assert parse_verdict("LEADS: yes\nPARAPHRASE: no\nLEANS: no\nWHY: it reaches the reply via latency.") == (True, "it reaches the reply via latency.")
    passed, why = parse_verdict("LEADS: Yes\nPARAPHRASE: no\nLEANS: **yes**\nWHY: the reply never uses the recollection.")
    assert not passed and why.startswith("it leans on the recollection") and "never uses" in why
    passed, why = parse_verdict("LEADS: no\nPARAPHRASE: yes\nLEANS: no")
    assert not passed and why == "it does not lead to the reply; it restates the reply"
    # decorated, numbered and bulleted checklists all parse
    assert parse_verdict("1. LEADS: yes\n2. PARAPHRASE: no\n3. LEANS: no\n4. WHY: ok")[0]
    assert parse_verdict("- **LEADS**: yes\n- **PARAPHRASE**: no\n- **LEANS**: no\n- **WHY**: ok") == (True, "ok")
    assert not parse_verdict("**LEADS:** no\n**PARAPHRASE:** no\n**LEANS:** no\n**WHY:** drifts")[0]
    # a bare verdict on the first line, decorated or labelled, still counts
    assert parse_verdict("**PASS**\nIt reaches the reply.")[0]
    assert parse_verdict("Verdict: PASS\nGood one.")[0]
    assert not parse_verdict("PASS/FAIL: FAIL\nno")[0] and not parse_verdict("FAIL: drifts")[0]
    # no verdict at all is not a rejection
    assert parse_verdict("") == (True, NO_VERDICT)
    assert parse_verdict("LEADS: yes") == (True, NO_VERDICT)  # a partial checklist is no checklist
    manager = PromptManager()
    with patch.object(PromptManager, "client") as client:
        client.chat.completions.create.return_value.choices[0].message.content = "LEADS: yes\nPARAPHRASE: no\nLEANS: no\nWHY: ok"
        assert manager.check_trace("q", "m", "r", "a", author="Sam", prev_answer="p") == (True, "ok")
        call = client.chat.completions.create.call_args.kwargs
        assert call["temperature"] == 0 and "LEANS:" in call["messages"][0]["content"]
        assert "for context:\np" in call["messages"][1]["content"]


def test_unjudged_traces_are_kept_and_counted():
    from raft.prompt_manager import NO_VERDICT

    manager = _manager()
    with patch.object(PromptManager, "reasoning_trace", return_value="Hm, fine."), \
         patch.object(PromptManager, "check_trace", return_value=(True, NO_VERDICT)), patch("raft.memories.hx.warn"):
        assert manager.reasoning_trace("q", "a", "", "") == "Hm, fine."
    assert MemoryManager.trace_stats["unjudged"] == 1 and MemoryManager.trace_stats["passed"] == 0


def test_a_helper_failure_is_a_rejection_not_a_crash():
    manager = _manager()
    with patch.object(PromptManager, "reasoning_trace", side_effect=[RuntimeError("502"), "Hm, ok."]) as write, \
         patch.object(PromptManager, "check_trace", return_value=(True, "fine")), \
         patch("raft.memories.hx.warn"), patch("raft.memories.hx.say"):
        assert manager.reasoning_trace("q", "a", "", "") == "Hm, ok."
    assert write.call_args_list[1].kwargs["objection"] == "the writer returned nothing"
    with patch.object(PromptManager, "reasoning_trace", return_value="Hm, ok."), \
         patch.object(PromptManager, "check_trace", side_effect=RuntimeError("timeout")), \
         patch("raft.memories.hx.warn"), patch("raft.memories.hx.say"):
        assert manager.reasoning_trace("q", "a", "", "") == ""
    assert MemoryManager.trace_stats["dropped"] == 1


def test_judge_receives_trace_and_reply_in_the_right_slots():
    manager = _manager()
    with patch.object(PromptManager, "reasoning_trace", return_value="TRACE"), \
         patch.object(PromptManager, "check_trace", return_value=(True, "ok")) as judge:
        manager.reasoning_trace("Q", "REPLY", "MEM", "PREV")
    assert judge.call_args.args == ("Q", "MEM", "TRACE", "REPLY") and judge.call_args.kwargs["prev_answer"] == "PREV"


def test_narrated_trace_is_rewritten_before_the_judge_is_asked():
    manager = _manager()
    with patch.object(PromptManager, "reasoning_trace", side_effect=["The commenter brings up decaf.", "Hm, decaf is bad."]) as write, \
         patch.object(PromptManager, "check_trace", return_value=(True, "ok")) as judge, patch("raft.memories.hx.say"):
        assert manager.reasoning_trace("q", "a", "", "") == "Hm, decaf is bad."
    assert judge.call_count == 1  # the narrated attempt never reached the judge
    assert "narrates" in write.call_args_list[1].kwargs["objection"] and "the commenter" in write.call_args_list[1].kwargs["objection"].lower()
    # on the last attempt the judge decides, so a legitimate mention does not cost the trace
    manager = _manager()
    with patch.object(PromptManager, "reasoning_trace", return_value="The commenter is right, and so am I."), \
         patch.object(PromptManager, "check_trace", return_value=(True, "ok")) as judge, patch("raft.memories.hx.say"):
        assert manager.reasoning_trace("q", "a", "", "") == "The commenter is right, and so am I."
    assert judge.call_count == 1


def test_recheck_regenerates_traces_with_recall_and_judges_the_rest(project, monkeypatch):
    project.finetune_path.parent.mkdir(parents=True, exist_ok=True)
    project.finetune_path.write_text(json.dumps([
        {"metadata": {"participants": {"q": "Pat", "a": "Sam"}, "date": "2024-05-01", "url": "u", "context": "an interview"}},
        {"example": {"question": "Why?", "answer": "Because.", "similar_memories": "from 2023: I said so.", "reasoning": "old-with-recall"}},
        {"example": {"question": "And?", "answer": "So.", "reasoning": "old-plain"}},
    ]))
    calls = []

    def trace(self, question, answer, memories, prev_answer, existing=""):
        calls.append((question, existing, prev_answer))
        return existing or "fresh"

    with patch.object(MemoryManager, "__init__", lambda self, *a, **k: None), patch.object(MemoryManager, "reasoning_trace", trace):
        generate_finetune.recheck_traces(project)
    assert calls == [("Why?", "", ""), ("And?", "old-plain", "Because.")]
    rows = json.loads(project.finetune_path.read_text())
    assert rows[1]["example"]["reasoning"] == "fresh" and rows[1]["example"]["reasoning_previous"] == "old-with-recall"
    assert rows[2]["example"]["reasoning"] == "old-plain" and "reasoning_previous" not in rows[2]["example"]
    # a trace no attempt could replace is kept aside, never lost
    with patch.object(MemoryManager, "__init__", lambda self, *a, **k: None), patch.object(MemoryManager, "reasoning_trace", return_value=""):
        generate_finetune.recheck_traces(project, regenerate="all")
    rows = json.loads(project.finetune_path.read_text())
    assert rows[1]["example"]["reasoning"] == "" and rows[1]["example"]["reasoning_previous"] == "fresh"
    assert not project.finetune_path.with_suffix(".json.tmp").exists()
    calls.clear()
    with patch.object(MemoryManager, "__init__", lambda self, *a, **k: None), patch.object(MemoryManager, "reasoning_trace", trace):
        generate_finetune.recheck_traces(project, regenerate="all")
    assert [c[1] for c in calls] == ["", ""]  # every trace written afresh
    monkeypatch.chdir(project.root)
    monkeypatch.setattr("sys.argv", ["raft", "ft:gen", "--rewrite-traces"])
    with patch.object(generate_finetune, "recheck_traces") as recheck, patch.object(oai_finetune, "create_openai_finetune_file") as convert:
        cli.main()
    assert recheck.call_args.kwargs["regenerate"] == "all"
    assert convert.call_args.kwargs["thinking"] is True
    assert state.load_meta(project)["thinking"] is True  # a re-check is a thinking dataset from now on


def test_recheck_runs_conversations_in_parallel(project, monkeypatch):
    project.finetune_path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for n in range(4):
        items.append({"metadata": {"participants": {"q": "Pat", "a": "Sam"}, "date": f"2024-0{n + 1}-01", "url": "u"}})
        items.append({"example": {"question": f"q{n}", "answer": f"a{n}", "reasoning": "old"}})
        items.append({"example": {"question": f"q{n}b", "answer": f"a{n}b", "reasoning": "old"}})
    project.finetune_path.write_text(json.dumps(items))
    monkeypatch.setenv("RAFT_WORKERS", "3")
    seen = []

    def trace(self, question, answer, memories, prev_answer, existing=""):
        seen.append((question, prev_answer))
        return "new"

    with patch.object(MemoryManager, "__init__", lambda self, *a, **k: None), patch.object(MemoryManager, "reasoning_trace", trace):
        stats = generate_finetune.recheck_traces(project, regenerate="all")
    rows = json.loads(project.finetune_path.read_text())
    assert all(r["example"]["reasoning"] == "new" and r["example"]["reasoning_previous"] == "old" for r in rows if "example" in r)
    assert sorted(seen) == sorted([(f"q{n}", "") for n in range(4)] + [(f"q{n}b", f"a{n}") for n in range(4)])
    assert isinstance(stats, dict)
