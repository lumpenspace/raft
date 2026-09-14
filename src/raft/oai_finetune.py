from openai import OpenAI
import os
import time
import json
import tiktoken
from typing import List, Dict, Any, Tuple, Union
from .prompt_manager import PromptManager
from .project import DatasetLike, dataset_paths
from openai.types.chat import ChatCompletionSystemMessageParam as SystemMessageParam

prompt_manager = PromptManager()

# Token budget of one training example (the exchange plus as much of the
# conversation before it as fits). 4096 was the 2023 finetuning window;
# every model raft targets now takes far more, so the default is 8192 and
# RAFT_MAX_EXAMPLE_TOKENS raises it further.
MAX_FINETUNE_LENGTH = int(os.environ.get("RAFT_MAX_EXAMPLE_TOKENS", "8192"))
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def count_tokens(prompt: object) -> int:
    """
    Count the number of tokens in a prompt.

    Args:
        prompt (object): The prompt to count tokens for.

    Returns:
        int: The number of tokens.
    """
    return len(encoding.encode(json.dumps(prompt), disallowed_special=()))


def recall_text(memories: str) -> str:
    """How a think block opens: what came to mind, or that nothing did."""
    memories = (memories or "").strip()
    return (
        f"Recalling what I have written before:\n{memories}"
        if memories
        else "Nothing I have written before bears on this directly."
    )


def think_block(example: Dict[str, Any]) -> str:
    """
    The persona's thinking before a reply: first the recall (the retrieved
    first-person summaries of earlier writing and conversations), then the
    reasoning from that recall to the reply.
    """
    recall = recall_text(example.get("similar_memories") or "")
    reasoning = (example.get("reasoning") or "").strip()
    body = f"{recall}\n\n{reasoning}" if reasoning else recall
    return f"<think>\n{body}\n</think>\n\n"


def oaify_example(
    example: Dict[str, Any], participants: Dict[str, str], thinking: bool = False
) -> Tuple[List[Union[Dict[str, Any], SystemMessageParam]], int]:
    """
    Convert an example to chat messages.

    Non-thinking format: the recalled memories arrive as a system note
    between the question and the reply. Thinking format: they are the
    opening of the reply's <think> block, followed by the reasoning trace,
    then the reply -- what the model is expected to do at inference.

    Returns:
        Tuple[List[Union[Dict[str, Any], SystemMessageParam]], int]:
        The converted example and its token count.
    """
    q_name = participants["q"]
    a_name = participants["a"]
    result: List[Union[Dict[str, Any], SystemMessageParam]] = [
        {
            "role": "user",
            "content": example["question"],
            "name": q_name.replace(" ", ""),
        }
    ]
    if thinking:
        answer = think_block(example) + example["answer"]
    else:
        answer = example["answer"]
        if "similar_memories" in example:
            result.append(
                SystemMessageParam(
                    role="system",
                    content=f"Relevant memories: {example['similar_memories']}",
                )
            )

    result.append(
        {
            "role": "assistant",
            "content": answer,
            "name": a_name.replace(" ", ""),
        }
    )
    return result, len(encoding.encode(json.dumps(result), disallowed_special=()))


def create_finetune_job(name: str, file: Any, model: str) -> Any:
    """
    Create a fine-tuning job.

    Args:
        name (str): The name of the fine-tuning job.
        file (Any): The file object to use for fine-tuning.
        model (str): The model to fine-tune.

    Returns:
        Any: The created fine-tuning job.
    """
    print(f"creating finetune job for: {model}")
    job = _get_client().fine_tuning.jobs.create(
        training_file=file.id, model=model, suffix=name
    )
    print(f"Fine tune started for job: {job.id} with model: {model}")
    return job


def launch_oai_finetune(dataset: DatasetLike, model: str) -> str:
    """
    Upload the dataset and create the fine-tuning job (without waiting).

    Returns:
        str: The job id, for wait_oai_finetune.
    """
    paths = dataset_paths(dataset)
    filename = paths.finetune_openai_path
    print(f"uploading file: {filename}")
    with open(file=filename, mode="rb") as source_file:
        file = _get_client().files.create(file=source_file, purpose="fine-tune")
    return create_finetune_job(paths.name, file, model).id


def wait_oai_finetune(job_id: str) -> str:
    """
    Poll a fine-tuning job until it finishes.

    Returns:
        str: The finetuned model id, or "" if the job failed.
    """
    status = ""
    while True:
        job = _get_client().fine_tuning.jobs.retrieve(job_id)
        if job.status in ["succeeded", "failed", "cancelled"]:
            print(f"Fine tune {job.status}. Model ID: {job.fine_tuned_model}")
            return job.fine_tuned_model or ""
        if status != job.status:
            print(f"Fine tune status: {job.status}")
            status = job.status
        time.sleep(2)


def run_oai_finetune(
    dataset: DatasetLike, model: str = "gpt-4o-mini-2024-07-18"
) -> str:
    """
    Run OpenAI fine-tuning for a given name, start to finish.

    Args:
        name (str): The name of the fine-tuning job.
        model (str): The OpenAI model to finetune.

    Returns:
        str: The finetuned model id, or "" if the job failed.
    """
    return wait_oai_finetune(launch_oai_finetune(dataset, model))


def create_openai_finetune_file(
    dataset: DatasetLike, type: str = "finetune", thinking: bool = False
) -> List[List[Union[Dict[str, Any], SystemMessageParam]]]:
    """
    Create the chat-format training file (OpenAI's jsonl shape, which the
    huggingface path reads too).

    Args:
        dataset: Dataset name or project paths.
        type (str, optional): "finetune" or "benchmark".
        thinking (bool): Put recall and reasoning in <think> blocks.

    Returns:
        List[List[Union[Dict[str, Any], SystemMessageParam]]]:
            The fine-tuning data.
    """
    paths = dataset_paths(dataset)
    input_path = (
        paths.finetune_path if type == "finetune" else paths.benchmark_generated_path
    )
    output_path = (
        paths.finetune_openai_path if type == "finetune" else paths.benchmark_openai_path
    )
    with input_path.open() as f:
        data = json.load(f)

    # Group the examples and reverse the order within each group
    groups: List[Dict[str, Any]] = []
    for item in data:
        if "metadata" in item:
            groups.append({"metadata": item["metadata"], "examples": []})
        elif "example" in item:
            groups[-1]["examples"].insert(0, item["example"])

    finetune_data: List[List[Union[Dict[str, Any], SystemMessageParam]]] = []
    for group in groups:
        meta = group["metadata"]
        system_message = prompt_manager.get_interview_system_message(
            questioner=meta["participants"]["q"],
            answerer=meta["participants"]["a"],
            date=meta["date"],
            context=meta.get("context", ""),
            thinking=thinking,
        )

        group_data: List[List[Union[Dict[str, Any], SystemMessageParam]]] = []

        for i, item in enumerate(group["examples"]):
            example: List[Union[Dict[str, Any], SystemMessageParam]]
            size: int
            example_size: int = count_tokens(system_message)
            example, size = oaify_example(item, meta["participants"], thinking)
            examples: List[Union[Dict[str, Any], SystemMessageParam]] = example
            example_size += size
            index = i
            if example_size < MAX_FINETUNE_LENGTH:
                while example_size < MAX_FINETUNE_LENGTH:
                    index = index + 1
                    if index >= len(group["examples"]):
                        break
                    older = group["examples"][index]

                    example, size = oaify_example(older, meta["participants"], thinking)
                    if example_size + size < MAX_FINETUNE_LENGTH:
                        examples = example + examples
                        example_size += size
            examples = [system_message] + examples
            group_data.append(examples)
        finetune_data = group_data + finetune_data

    # Save the fine-tuned data to a new JSONL file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        finetune_data.reverse()
        for item in finetune_data:
            f.write(json.dumps({"messages": item}))
            f.write("\n")
    return finetune_data
