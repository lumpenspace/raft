import json
import time
from .memories import MemoryManager
from .files_helper import begin_json_file, end_json_file, write_context_to_file
def process_transcripts(name: str, suffix: str, is_benchmark: bool = False):

    with open(f"data/{name}_transcript_{suffix}.json") as f:
        interview_data = json.load(f)

    target_filename = f"{name}_{'benchmark' if is_benchmark else 'finetune'}.json"
    index = suffix if isinstance(suffix, int) else 1
    # Extract metadata
    metadata = {key: interview_data[key] for key in ["participants", "date", "url"]}
    memory_manager = MemoryManager(name, metadata)

    # Write metadata to file
    write_context_to_file(target_filename, {"metadata": metadata}, index, 0)
    prev_answer = None

    # For each exchange in the interview
    for j, exchange in enumerate(interview_data["exchanges"]):
        question, answer = exchange

        context = {
            "question": question,
            "answer": answer
        }

        similar_memories = memory_manager.get_similar_and_summarize(exchange, prev_answer)
        if len(similar_memories) > 0:
            context["similar_memories"] = similar_memories

        print(suffix)
        print(isinstance(suffix, int))
        write_context_to_file(
            target_filename, {"example": context},
            index,
            j+1
        )

        prev_answer = answer

def generate_finetune(name: str):
    begin_json_file(f"{name}_finetune")
    i = 1
    while True:
        try:
            print(f"processing transcript #{i}")
            process_transcripts(name, i)
        except FileNotFoundError:
            print(f"file #{i} not found")
            break
        time.sleep(2)
        i += 1
        
    end_json_file(name)
    print(f"Generic finetune file generated in: data/{name}_finetune.json")

def generate_benchmark(name: str):
    begin_json_file(f"{name}_benchmark")
    process_transcripts(name, 'benchmark', True)
    end_json_file(f"{name}_benchmark")
    print("Done!")