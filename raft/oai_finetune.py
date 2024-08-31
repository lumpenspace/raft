from openai import OpenAI
import time
import json
import tiktoken
from typing import List, Dict, Any, Tuple, Union
from .prompt_manager import PromptManager
from openai.types.chat import ChatCompletionSystemMessageParam as SystemMessageParam

prompt_manager = PromptManager()

MAX_FINETUNE_LENGTH = 4096
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
client = OpenAI()


def count_tokens(prompt: object) -> int:
    """
    Count the number of tokens in a prompt.

    Args:
        prompt (object): The prompt to count tokens for.

    Returns:
        int: The number of tokens.
    """
    return len(encoding.encode(json.dumps(prompt)))


def oaify_example(
    example: Dict[str, Any], participants: Dict[str, str]
) -> Tuple[List[Union[Dict[str, Any], SystemMessageParam]], int]:
    """
    Convert an example to OpenAI format.

    Args:
        example (Dict[str, Any]): The example to convert.
        participants (Dict[str, str]): The participants information.

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
            "content": example["answer"],
            "name": a_name.replace(" ", ""),
        }
    )
    return result, len(encoding.encode(json.dumps(result)))


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
    job = client.fine_tuning.jobs.create(
        training_file=file.id, model=model, suffix=name
    )
    print(f"Fine tune started for job: {job.id} with model: {model}")
    return job


def run_oai_finetune(name: str) -> None:
    """
    Run OpenAI fine-tuning for a given name.

    Args:
        name (str): The name of the fine-tuning job.
    """
    models = ["gpt-3.5-turbo"]
    status_dict: Dict[str, str] = {model: "" for model in models}
    filename = f"data/{name}_finetune_openai.jsonl"

    print(f"uploading file: {filename}")
    with open(file=filename, mode="rb") as source_file:
        file = client.files.create(file=source_file, purpose="fine-tune")

    for model in models:
        job = create_finetune_job(name, file, model)

        while True:
            job_update = client.fine_tuning.jobs.retrieve(job.id)
            if job_update.status in ["succeeded", "failed"]:
                print(
                    f"Fine tune completed for model {model}",
                    f"Model ID: {job_update.fine_tuned_model}",
                )
                status_dict[model] = job_update.status
                break
            if status_dict[model] != job_update.status:
                print(f"Fine tune status for model {model}:")
                print(job_update.status)
                status_dict[model] = job_update.status
            time.sleep(2)


def create_openai_finetune_file(
    name: str, type: str = "finetune"
) -> List[List[Union[Dict[str, Any], SystemMessageParam]]]:
    """
    Create an OpenAI fine-tuning file.

    Args:
        name (str): The name of the fine-tuning job.
        type (str, optional): The type of file to create. Default: "finetune".

    Returns:
        List[List[Union[Dict[str, Any], SystemMessageParam]]]:
            The fine-tuning data.
    """
    with open(f"data/{name}_{type}.json") as f:
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
        )

        group_data: List[List[Union[Dict[str, Any], SystemMessageParam]]] = []

        for i, item in enumerate(group["examples"]):
            print(i)
            example: List[Union[Dict[str, Any], SystemMessageParam]]
            size: int
            example_size: int = count_tokens(system_message)
            example, size = oaify_example(item, meta["participants"])
            examples: List[Union[Dict[str, Any], SystemMessageParam]] = example
            example_size += size
            index = i
            if example_size < MAX_FINETUNE_LENGTH:
                while example_size < MAX_FINETUNE_LENGTH:
                    index = index + 1
                    print("index", index)
                    if index >= len(group["examples"]):
                        break
                    older = group["examples"][index]

                    example, size = oaify_example(older, meta["participants"])
                    if example_size + size < MAX_FINETUNE_LENGTH:
                        examples = example + examples
                        example_size += size
            examples = [system_message] + examples
            group_data.append(examples)
        finetune_data = group_data + finetune_data

    # Save the fine-tuned data to a new JSONL file
    with open(f"data/{name}_{type}_openai.jsonl", "w") as f:
        finetune_data.reverse()
        for item in finetune_data:
            f.write(json.dumps({"messages": item}))
            f.write("\n")
    return finetune_data
