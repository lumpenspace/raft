"""
Turn the conversations (transcripts) into finetune examples, each
exchange augmented with what the persona could recall at the time: the
retrieved, first-person summaries of earlier writing and earlier
conversations -- and, for thinking models, the reasoning that leads from
that recall to the reply actually given.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from . import hx
from .memories import PACE, MemoryManager, MetaDataKeyEnum
from .files_helper import begin_json_file, end_json_file, write_context_to_file
from .project import DatasetLike, dataset_paths


def process_transcripts(
    dataset: DatasetLike, suffix: str, is_benchmark: bool, thinking: bool = False
) -> None:
    """
    Process one conversation and append its examples to the generic file.

    Args:
        dataset: Dataset name or project paths.
        suffix (str): The transcript index, or "benchmark".
        is_benchmark (bool): Benchmark conversations are never remembered
            as conversation memories (they would leak into training).
        thinking (bool): Also write a reasoning trace per exchange.
    """
    paths = dataset_paths(dataset)
    with paths.transcript_path(suffix).open() as f:
        interview_data = json.load(f)

    target_file = paths.benchmark_generated_path if is_benchmark else paths.finetune_path
    index = int(suffix) if str(suffix).isdigit() else 1
    metadata: dict[MetaDataKeyEnum, Any] = {
        MetaDataKeyEnum(key): interview_data[key]
        for key in ["participants", "date", "url"]
    }
    memory_manager = MemoryManager(paths, metadata)

    header = {key.value: value for key, value in metadata.items()}
    if interview_data.get("context"):
        header["context"] = interview_data["context"]
    write_context_to_file(target_file, {"metadata": header}, index, 0)
    prev_answer = ""

    for j, exchange in enumerate(interview_data["exchanges"]):
        question, answer = exchange

        context = {"question": question, "answer": answer}

        similar_memories = memory_manager.get_similar_and_summarize(
            exchange, prev_answer, store=not is_benchmark
        )
        if len(similar_memories) > 0:
            context["similar_memories"] = similar_memories
        if thinking:
            context["reasoning"] = memory_manager.reasoning_trace(question, answer, similar_memories, prev_answer)

        write_context_to_file(target_file, {"example": context}, index, j + 1)

        prev_answer = answer


def generate_finetune(dataset: DatasetLike, thinking: bool = False) -> None:
    """
    Generate the generic finetune file from every conversation, in order.

    Args:
        dataset: Dataset name or project paths.
        thinking (bool): Write reasoning traces for a thinking model.
    """
    paths = dataset_paths(dataset)
    MemoryManager.reset_trace_stats()
    begin_json_file(paths.finetune_path)
    i = 1
    while True:
        try:
            hx.step(f"conversation #{i}")
            process_transcripts(paths, f"{i}", False, thinking=thinking)
        except FileNotFoundError:
            hx.say(f"{i - 1} conversation(s) processed")
            break
        if PACE:
            time.sleep(PACE)
        i += 1

    end_json_file(paths.finetune_path)
    if thinking:
        hx.say(f"reasoning traces: {MemoryManager.trace_stats}")
    hx.ok(f"generic finetune file generated in: {paths.finetune_path}")


def _write_generic(path: Path, items: List[Any]) -> None:
    """Replace the generic file atomically (a temp file, then a rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(items, f, indent=4)
    os.replace(tmp, path)


def recheck_traces(dataset: DatasetLike, regenerate: str = "recall") -> Dict[str, int]:
    """
    Judge every reasoning trace in the generic file against its reply and
    rewrite the ones that fail, without redoing retrieval. A trace that no
    attempt could replace is kept aside as "reasoning_previous" (never
    lost) while "reasoning" is emptied to recall-only; the file is written
    after every conversation, atomically, so an interrupted run keeps its
    progress. Conversations are independent, so RAFT_WORKERS of them run
    at once (default 1).

    Args:
        regenerate: which existing traces to discard before judging --
            "none" (judge all as they are), "recall" (examples that carry
            recall, whose traces predate the rule about leaning on it only
            as far as the reply does), or "all" (a new writer, say).
    """
    if regenerate not in ("none", "recall", "all"):
        raise ValueError(f"regenerate must be none, recall or all, not {regenerate!r}")
    paths = dataset_paths(dataset)
    try:
        with paths.finetune_path.open() as f:
            items = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{paths.finetune_path} is not complete JSON ({e}); a run may still be writing it -- "
            f"finish or rerun `{paths.command('ft:gen')}` first"
        ) from e
    MemoryManager.reset_trace_stats()

    # Group the flat file into conversations: a metadata item and its examples.
    groups: List[tuple] = []
    for item in items:
        if "metadata" in item:
            groups.append((item["metadata"], []))
        elif groups and item.get("example", {}).get("answer"):
            groups[-1][1].append(item["example"])

    write_lock = threading.Lock()
    done = {"conversations": 0, "checked": 0}

    def process(group: tuple) -> int:
        meta, examples = group
        manager = MemoryManager(paths, {MetaDataKeyEnum(k): meta[k] for k in ("participants", "date", "url") if k in meta})
        prev_answer = ""
        for example in examples:
            memories = example.get("similar_memories", "")
            previous = example.get("reasoning", "")
            discard = regenerate == "all" or (regenerate == "recall" and memories)
            hx.step(" ".join(example["question"].split())[:100])
            trace = manager.reasoning_trace(
                example["question"], example["answer"], memories, prev_answer, existing="" if discard else previous
            )
            if previous and trace != previous:
                example["reasoning_previous"] = previous
            example["reasoning"] = trace
            prev_answer = example["answer"]
        with write_lock:
            done["conversations"] += 1
            done["checked"] += len(examples)
            _write_generic(paths.finetune_path, items)
            hx.say(f"  {done['conversations']}/{len(groups)} conversation(s) re-judged")
        return len(examples)

    workers = max(1, int(os.environ.get("RAFT_WORKERS", "1")))
    if workers == 1:
        for group in groups:
            process(group)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(process, groups))
    _write_generic(paths.finetune_path, items)
    hx.ok(f"{done['checked']} reasoning trace(s) checked: {MemoryManager.trace_stats}")
    return dict(MemoryManager.trace_stats)


def generate_benchmark(dataset: DatasetLike, thinking: bool = False) -> None:
    """
    Generate benchmark data for a given dataset.

    Args:
        dataset: Dataset name or project paths.
        thinking (bool): Write reasoning traces for a thinking model.
    """
    paths = dataset_paths(dataset)
    begin_json_file(paths.benchmark_generated_path)
    process_transcripts(paths, "benchmark", True, thinking=thinking)
    end_json_file(paths.benchmark_generated_path)
    hx.ok(f"benchmark file generated in: {paths.benchmark_generated_path}")
