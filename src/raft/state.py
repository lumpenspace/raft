"""
Per-dataset state: what exists on disk, and the sidecar meta file.

data/{name}_meta.json records what the pipeline itself cannot recover
from its artifacts -- the target's name, the finetuned model ids, the
test questions collected while a job ran -- so `raft interactive` can
resume a dataset where it left off.
"""

import glob
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

# data/{name}<suffix> files that are pipeline artifacts, not corpora.
ARTIFACT_SUFFIXES = (
    "_chunked",
    "_finetune",
    "_finetune_openai",
    "_benchmark",
    "_benchmark_openai",
)


def meta_path(name: str) -> str:
    """The dataset's meta file path."""
    return f"data/{name}_meta.json"


def load_meta(name: str) -> Dict[str, Any]:
    """Load the dataset meta, folding in the legacy model_id key."""
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path(name)):
        with open(meta_path(name)) as f:
            meta = json.load(f)
    legacy = meta.pop("model_id", None)
    if legacy and not any(
        m.get("model") == legacy for m in meta.get("finetuned_models", [])
    ):
        meta.setdefault("finetuned_models", []).insert(
            0, {"model": legacy, "backend": "openai", "date": ""}
        )
    return meta


def save_meta(name: str, meta: Dict[str, Any]) -> None:
    """Write the dataset meta file."""
    os.makedirs("data", exist_ok=True)
    with open(meta_path(name), "w") as f:
        json.dump(meta, f, indent=2)


def update_meta(name: str, **fields: Any) -> Dict[str, Any]:
    """Merge fields into the dataset meta and save it."""
    meta = load_meta(name)
    meta.update(fields)
    save_meta(name, meta)
    return meta


def record_finetuned_model(name: str, model: str, backend: str) -> None:
    """Remember a finetuned model (an OpenAI ft id, or a local adapter path)."""
    meta = load_meta(name)
    models = meta.setdefault("finetuned_models", [])
    if not any(m.get("model") == model for m in models):
        models.append(
            {
                "model": model,
                "backend": backend,
                "date": datetime.now(timezone.utc).date().isoformat(),
            }
        )
    save_meta(name, meta)


def finetuned_model(name: str) -> str:
    """The most recently recorded finetuned model, or an empty string."""
    models = load_meta(name).get("finetuned_models", [])
    return models[-1]["model"] if models else ""


def model_backend(name: str, model: str = "") -> str:
    """
    The recorded backend ("openai" or "hf") for a model -- the newest
    entry, or the one matching `model`. Empty if never recorded.
    """
    for entry in reversed(load_meta(name).get("finetuned_models", [])):
        if not model or entry.get("model") == model:
            return entry.get("backend", "")
    return ""


def test_questions(name: str) -> List[str]:
    """The test questions collected for this dataset."""
    return list(load_meta(name).get("test_questions", []))


def add_test_question(name: str, question: str) -> None:
    """Store a test question (deduplicated)."""
    meta = load_meta(name)
    questions = meta.setdefault("test_questions", [])
    if question not in questions:
        questions.append(question)
    save_meta(name, meta)


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def dataset_status(name: str) -> Dict[str, Any]:
    """
    What exists on disk for a dataset, one key per pipeline artifact.
    """
    # convo_structurer owns the transcript naming scheme; counting via
    # its next-free-index helper keeps a single owner for the pattern.
    from .convo_structurer import next_transcript_index

    meta = load_meta(name)
    return {
        "corpus_docs": _count_lines(f"data/{name}.jsonl"),
        "chunks": _count_lines(f"data/{name}_chunked.jsonl"),
        "embedded": os.path.exists(f"data/{name}/chroma.sqlite3"),
        "transcripts": next_transcript_index(name) - 1,
        "finetune_file": os.path.exists(f"data/{name}_finetune.json"),
        "openai_file": os.path.exists(f"data/{name}_finetune_openai.jsonl"),
        "model": finetuned_model(name),
        "benchmark": os.path.exists(f"data/{name}_transcript_benchmark.json"),
        "questions": len(meta.get("test_questions", [])),
        "evaluated": bool(meta.get("evaluated_at")),
    }


def list_datasets() -> List[str]:
    """
    Dataset names found under data/ (from corpora, transcripts and meta
    files, minus pipeline-artifact suffixes).
    """
    names = set()
    for path in glob.glob("data/*.jsonl"):
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.endswith(ARTIFACT_SUFFIXES):
            names.add(stem)
    for path in glob.glob("data/*_meta.json"):
        names.add(os.path.basename(path)[: -len("_meta.json")])
    for path in glob.glob("data/*_transcript_1.json"):
        names.add(os.path.basename(path)[: -len("_transcript_1.json")])
    return sorted(names)
