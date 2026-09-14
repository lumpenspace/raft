"""Offline regression coverage for project paths and source routing."""
import json
from unittest.mock import patch

import pytest

from raft import cli, flows, generate_finetune, oai_finetune, sources, state
from raft.convo_structurer import write_transcript
from raft.project import DatasetPaths, ProjectError, find_project, initialize_project
from raft.tweet_mode import import_documents


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return initialize_project(tmp_path / "persona")[0]


def tweet(role=None):
    return {"id": "tweet:1", "text": "Pat: Why?\nSam: Because.",
            "metadata": {"target_url": "https://example.com/1", **({"dataset_role": role} if role else {})},
            "messages": [{"username": "pat", "text": "Why?"}, {"username": "sam", "text": "Because."}]}


def test_projects_discover_from_subdirectories_and_preserve_state(project):
    state.update_meta(project, target="Sam")
    reopened, created = initialize_project(project.root)
    assert not created
    assert state.load_meta(reopened)["target"] == "Sam"
    assert find_project(project.root / "corpus").root == project.root
    assert not (project.root / "data").exists()


def test_init_rejects_populated_reserved_directory(tmp_path):
    (tmp_path / "corpus").mkdir()
    valuable = tmp_path / "corpus" / "existing.txt"
    valuable.write_text("keep")
    with pytest.raises(ProjectError):
        initialize_project(tmp_path)
    assert valuable.read_text() == "keep"
    assert not (tmp_path / "raft.json").exists()


@pytest.mark.parametrize("role,corpus,transcripts", [("auto", 0, 1), ("conversation", 0, 1), ("corpus", 1, 0)])
def test_tweet_source_destination(project, role, corpus, transcripts):
    import_documents(project, [tweet()], "sam", role=role)
    result = state.dataset_status(project)
    assert result["corpus_docs"] == corpus
    assert result["transcripts"] == transcripts


def test_explicit_ariadne_roles_are_respected(project):
    import_documents(project, [tweet("corpus")], "sam")
    assert state.dataset_status(project)["corpus_docs"] == 1
    assert state.dataset_status(project)["transcripts"] == 0


def test_source_plan_accepts_multiple_sources_and_roles():
    with patch("raft.flows.choose", side_effect=[0, 0, 1, 1, 1, 0, 8]):
        assert flows.plan_sources() == [
            {"kind": 0, "role": "conversation"}, {"kind": 1, "role": "corpus"},
            {"kind": 1, "role": "conversation"}]


def test_fetched_interview_never_enters_grounding(project):
    def fetch(destination, url):
        return sources.append_corpus_records(destination, [{"content": "Why? Because.", "date": "2024-01-01", "link": url}])
    with patch("raft.flows.ask", return_value="https://example.com/interview"), \
         patch("raft.sources.fetch_url", side_effect=fetch), \
         patch("raft.convo_structurer.structure_raw_conversation", return_value={
             "participants": {"q": "Pat", "a": "Sam"}, "exchanges": [["Why?", "Because."]]}):
        flows.import_planned_source(project, "Sam", {"kind": 3, "role": "conversation"})
    assert not project.corpus_path.exists()
    transcript = json.loads(project.transcript_path(1).read_text())
    assert transcript["url"] == "https://example.com/interview"
    assert transcript["date"] == "2024-01-01"
    assert transcript["exchanges"] == [["Why?", "Because."]]


@pytest.mark.parametrize("legacy", [False, True])
def test_conversations_only_prep_generates_actual_training_file(project, legacy):
    dataset = DatasetPaths.legacy("sam", project.root) if legacy else project
    write_transcript(dataset, {"q": "Pat", "a": "Sam"}, "2024-01-01", "", [["Why?", "Because."]])
    with patch("raft.flows.confirm", side_effect=lambda prompt, default=True: "thinking" not in prompt), \
         patch("raft.files_helper.chunker") as chunk, \
         patch("raft.embeddings_helpers.store_grounding_embeddings") as embed, \
         patch("raft.memories.OpenAI") as client, \
         patch("raft.generate_finetune.time.sleep"):
        flows.phase_prep(dataset)
    chunk.assert_not_called()
    embed.assert_not_called()
    client.return_value.chat.completions.create.assert_not_called()
    examples = [json.loads(row) for row in dataset.finetune_openai_path.read_text().splitlines()]
    assert examples[0]["messages"][-1]["content"] == "Because."
    assert [m["role"] for m in examples[0]["messages"]] == ["system", "user", "assistant"]
    assert not dataset.chroma_path.exists()


def test_cli_resolves_project_without_dataset_name(project, monkeypatch):
    monkeypatch.chdir(project.root / "conversations")
    monkeypatch.setattr("sys.argv", ["raft", "ft:gen", "--generic"])
    with patch.object(generate_finetune, "generate_finetune") as generate:
        cli.main()
    assert generate.call_args.args[0].root == project.root


def test_cli_explicit_name_preserves_legacy_layout(project, monkeypatch):
    monkeypatch.chdir(project.root)
    monkeypatch.setattr("sys.argv", ["raft", "ft:gen", "old", "--oai"])
    with patch.object(oai_finetune, "create_openai_finetune_file") as generate:
        cli.main()
    assert generate.call_args.args[0].corpus_path == project.root / "data" / "old.jsonl"


def test_cli_init(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["raft", "init", str(tmp_path / "new")])
    cli.main()
    assert (tmp_path / "new" / "raft.json").exists()


def test_serving_without_grounding_sends_plain_chat(project):
    from raft.memories import MemoryManager
    with patch("raft.memories.OpenAI") as client:
        client.return_value.chat.completions.create.return_value.choices[0].message.content = "Because."
        manager = MemoryManager(project, {})
        assert manager.ask_question("Why?", model="ft:test") == "Because."
    messages = client.return_value.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert not project.chroma_path.exists()


def test_fresh_interactive_session_asks_sources_before_phases(project):
    calls = []
    with patch("raft.flows.ask", return_value="Sam"), \
         patch("raft.flows.phase_gather", side_effect=lambda *args: calls.append("sources")), \
         patch("raft.flows.choose", side_effect=lambda *args, **kwargs: calls.append("phase") or 5):
        flows.run_interactive(project)
    assert calls == ["sources", "phase"]
