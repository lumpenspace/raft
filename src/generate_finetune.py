from src.memories import MemoryManager
from prompts import summarize_memory
import json
import tiktoken
from prompts import summarize_memory

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(prompt: object):
    len(encoding.encode(json.dumps(prompt)))

def oaify_example(example: dict, participants:dict):
    q_name = participants['q']
    a_name = participants['a']
    result = []
    result.append({
        'role': 'user',
        'content': example['question'],
        'name': q_name
    })
    if (example.has_key('similar_memories')):
       result.append({
          'role': 'function',
          'content': example['similar_memories'],
          'name': 'retrieve_memories'
      })

    result.append({
        'role': 'assistant',
        'content': example['answer'],
        'name': a_name
    })
    return result, len(encoding.encode(json.dumps(result)));
    
def create_openai_finetune(name: str):
    with open(f"data/{name}_finetune.json") as f:
        data = json.load(f)

    # group the examples and reverse the order within each group
    groups = []
    for item in data:
        if 'metadata' in item:
            groups.append({ "metadata": item['metadata'], "examples": [] })
        elif 'example' in item:
            groups[-1]['examples'].prepend(item['example'])
        

    for group in groups:
        meta = group['metadata']
        system_message = {
            'role': 'system',
            'content': f"This is an interview between {meta['participants']['q']} and {meta['participants']['a']} on {meta['date']}."
        }
        group['examples'].insert(0, system_message)

        for item in group["examples"]:
          example_tokens = count_tokens(item['example'])
          example = item['example']
          example, size = oaify_example(example, meta['participants'])

          # Prepend the example
          fine_tune_data.insert(0, example)
          total_tokens += size

    # Save the fine-tuned data to a new JSONL file
    with open('data/openai_finetune.jsonl', 'w') as f:
        for item in fine_tune_data:
            f.write(json.dumps(item) + '\n')

def generate_finetune(name: str):
    memory_manager = MemoryManager(name)

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

        # Write metadata to file
        write_context_to_file(name, {"metadata": metadata}, i, 0)

        # For each exchange in the interview
        for j, exchange in enumerate(interview_data["exchanges"]):
            question, answer = exchange

            # Retrieve similar memories
            similar_memories = memory_manager.get_similar_and_summarize(question, prev_answer)

            context = {
                "question": question,
                "answer": answer
            }
            if len(similar_memories) > 0:
                context["similar_memories"] = similar_memories

            write_context_to_file(name, {"example": context}, i, j+1)

            # Update previous question and answer
            prev_answer = answer

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