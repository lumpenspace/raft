"""
Per-dataset state: what exists on disk, and the sidecar meta file.

data/{name}_meta.json records what the pipeline itself cannot recover
from its artifacts -- the target's name, the finetuned model ids, the
test questions collected while a job ran -- so `raft interactive` can
resume a dataset where it left off.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .project import DatasetLike, dataset_paths

# data/{name}<suffix> files that are pipeline artifacts, not corpora.
ARTIFACT_SUFFIXES = (
    "_chunked",
    "_finetune",
    "_finetune_openai",
    "_benchmark",
    "_benchmark_openai",
)


def meta_path(dataset: DatasetLike) -> str:
    """The dataset's meta file path."""
    return str(dataset_paths(dataset).meta_path)


def load_meta(dataset: DatasetLike) -> Dict[str, Any]:
    """Load the dataset meta, folding in the legacy model_id key."""
    paths = dataset_paths(dataset)
    meta: Dict[str, Any] = {}
    if paths.meta_path.exists():
        with paths.meta_path.open() as f:
            meta = json.load(f)
    if paths.project and paths.manifest:
        target = paths.manifest.get("target")
        if target and not meta.get("target"):
            meta["target"] = target
    legacy = meta.pop("model_id", None)
    if legacy and not any(
        m.get("model") == legacy for m in meta.get("finetuned_models", [])
    ):
        meta.setdefault("finetuned_models", []).insert(
            0, {"model": legacy, "backend": "openai", "date": ""}
        )
    return meta


def save_meta(dataset: DatasetLike, meta: Dict[str, Any]) -> None:
    """Write the dataset meta file."""
    path = dataset_paths(dataset).meta_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(meta, f, indent=2)


def update_meta(dataset: DatasetLike, **fields: Any) -> Dict[str, Any]:
    """Merge fields into the dataset meta and save it."""
    meta = load_meta(dataset)
    meta.update(fields)
    save_meta(dataset, meta)
    return meta


def record_finetuned_model(dataset: DatasetLike, model: str, backend: str) -> None:
    """Remember a finetuned model (an OpenAI ft id, or a local adapter path)."""
    meta = load_meta(dataset)
    models = meta.setdefault("finetuned_models", [])
    if not any(m.get("model") == model for m in models):
        models.append(
            {
                "model": model,
                "backend": backend,
                "date": datetime.now(timezone.utc).date().isoformat(),
            }
        )
    save_meta(dataset, meta)


def finetuned_model(dataset: DatasetLike) -> str:
    """The most recently recorded finetuned model, or an empty string."""
    models = load_meta(dataset).get("finetuned_models", [])
    return models[-1]["model"] if models else ""


def model_backend(dataset: DatasetLike, model: str = "") -> str:
    """
    The recorded backend ("openai" or "hf") for a model -- the newest
    entry, or the one matching `model`. Empty if never recorded.
    """
    for entry in reversed(load_meta(dataset).get("finetuned_models", [])):
        if not model or entry.get("model") == model:
            return entry.get("backend", "")
    return ""


def test_questions(dataset: DatasetLike) -> List[str]:
    """The test questions collected for this dataset."""
    return list(load_meta(dataset).get("test_questions", []))


def add_test_question(dataset: DatasetLike, question: str) -> None:
    """Store a test question (deduplicated)."""
    meta = load_meta(dataset)
    questions = meta.setdefault("test_questions", [])
    if question not in questions:
        questions.append(question)
    save_meta(dataset, meta)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def dataset_status(dataset: DatasetLike) -> Dict[str, Any]:
    """
    What exists on disk for a dataset, one key per pipeline artifact.
    """
    # convo_structurer owns the transcript naming scheme; counting via
    # its next-free-index helper keeps a single owner for the pattern.
    from .convo_structurer import next_transcript_index

    paths = dataset_paths(dataset)
    meta = load_meta(paths)
    return {
        "corpus_docs": _count_lines(paths.corpus_path),
        "chunks": _count_lines(paths.chunks_path),
        "embedded": (paths.chroma_path / "chroma.sqlite3").exists(),
        "transcripts": next_transcript_index(paths) - 1,
        "finetune_file": paths.finetune_path.exists(),
        "openai_file": paths.finetune_openai_path.exists(),
        "model": finetuned_model(paths),
        "benchmark": paths.transcript_path("benchmark").exists(),
        "questions": len(meta.get("test_questions", [])),
        "evaluated": bool(meta.get("evaluated_at")),
    }


def list_datasets() -> List[str]:
    """
    Dataset names found under data/ (from corpora, transcripts and meta
    files, minus pipeline-artifact suffixes).
    """
    names = set()
    data = Path("data")
    for path in data.glob("*.jsonl"):
        stem = path.stem
        if not stem.endswith(ARTIFACT_SUFFIXES):
            names.add(stem)
    for path in data.glob("*_meta.json"):
        names.add(path.name[: -len("_meta.json")])
    for path in data.glob("*_transcript_1.json"):
        names.add(path.name[: -len("_transcript_1.json")])
    return sorted(names)
