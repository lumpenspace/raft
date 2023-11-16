from src.memories import MemoryManager
import json
import time

def generate_finetune(name: str):

    begin_file(name)

    prev_answer = None

    # Loop over multiple transcript files
    i = 1
    while True:
        try:
            # Load the interview data from the file related to the given name
            with open(f"data/{name}_transcript_{i}.json") as f:
                interview_data = json.load(f)
        except FileNotFoundError:
            break

        # Extract metadata
        metadata = {key: interview_data[key] for key in ["participants", "date", "url"]}
        memory_manager = MemoryManager(name, metadata)

        # Write metadata to file
        write_context_to_file(name, {"metadata": metadata}, i, 0)

        # For each exchange in the interview
        for j, exchange in enumerate(interview_data["exchanges"]):
            question, answer = exchange

            # Retrieve similar memories
            similar_memories = memory_manager.get_similar_and_summarize(exchange, prev_answer)

            context = {
                "question": question,
                "answer": answer
            }
            if len(similar_memories) > 0:
                context["similar_memories"] = similar_memories

            write_context_to_file(name, {"example": context}, i, j+1)

            # Update previous question and answer
            prev_answer = answer
            # wait 2 seconds
            time.sleep(2)

        i += 1

    # Open the output file in append mode and write the closing bracket for the JSON array
    end_file(name)
    print("Done!")

def begin_file(name: str):
    # Open the output file and write the opening bracket for the JSON array
    write_to_file(name, '[\n', 'w')

def end_file(name: str):
    # Open the output file in append mode and write the closing bracket for the JSON array
    write_to_file(name, '\n]', 'a')

def write_to_file(name: str, data: str, mode: str = 'a'):
    with open(f"data/{name}_finetune.json", mode) as f:
        f.write(data)

def write_context_to_file(name: str, context: dict, i: int, j: int):
    # Open the output file in append mode and write the context
    with open(f"data/{name}_finetune.json", 'a') as f:
        # If it's not the first item, prepend a comma
        if i > 1 or j > 0:
            f.write(',\n')
        json.dump(context, f, indent=4)