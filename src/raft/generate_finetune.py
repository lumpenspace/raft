import json
import time
from typing import Any
from .memories import MemoryManager, MetaDataKeyEnum
from .files_helper import begin_json_file, end_json_file, write_context_to_file
from .project import DatasetLike, dataset_paths


def process_transcripts(
    dataset: DatasetLike, suffix: str, is_benchmark: bool
) -> None:
    """
    Process transcripts and generate fine-tuning data.

    Args:
        name (str): The name of the dataset.
        suffix (str): The suffix for the transcript file.
        is_benchmark (bool, optional): Whether this is for benchmarking.
            Defaults to False.
        no_useful_check (bool, optional): Whether to skip the usefulness check.
        Defaults to False.
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

    write_context_to_file(
        target_file,
        {"metadata": {key.value: value for key, value in metadata.items()}},
        index,
        0,
    )
    prev_answer = ""

    for j, exchange in enumerate(interview_data["exchanges"]):
        question, answer = exchange

        context = {"question": question, "answer": answer}

        similar_memories = memory_manager.get_similar_and_summarize(
            exchange, prev_answer
        )
        if len(similar_memories) > 0:
            context["similar_memories"] = similar_memories

        write_context_to_file(target_file, {"example": context}, index, j + 1)

        prev_answer = answer


def generate_finetune(dataset: DatasetLike) -> None:
    """
    Generate fine-tuning data for a given dataset.

    Args:
        name (str): The name of the dataset.
    """
    paths = dataset_paths(dataset)
    begin_json_file(paths.finetune_path)
    i = 1
    while True:
        try:
            print(f"processing transcript #{i}")
            process_transcripts(paths, f"{i}", False)
        except FileNotFoundError:
            print(f"file #{i} not found")
            break
        time.sleep(2)
        i += 1

    end_json_file(paths.finetune_path)
    print(f"Generic finetune file generated in: {paths.finetune_path}")


def generate_benchmark(dataset: DatasetLike) -> None:
    """
    Generate benchmark data for a given dataset.

    Args:
        name (str): The name of the dataset.
    """
    paths = dataset_paths(dataset)
    begin_json_file(paths.benchmark_generated_path)
    process_transcripts(paths, "benchmark", True)
    end_json_file(paths.benchmark_generated_path)
    print("Done!")
