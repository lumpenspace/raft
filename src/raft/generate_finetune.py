"""
Turn the conversations (transcripts) into finetune examples, each
exchange augmented with what the persona could recall at the time: the
retrieved, first-person summaries of earlier writing and earlier
conversations -- and, for thinking models, the reasoning that leads from
that recall to the reply actually given.
"""

import json
import time
from typing import Any

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
    hx.ok(f"generic finetune file generated in: {paths.finetune_path}")


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
