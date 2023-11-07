from src.memories import MemoryManager
from prompts import summarize_memory
import json
import tiktoken
from prompts import summarize_memory

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(prompt: object):
    return len(encoding.encode(json.dumps(prompt)))

def oaify_example(example: dict, participants:dict):
    q_name = participants['q']
    a_name = participants['a']
    result = []
    result.append({
        'role': 'user',
        'content': example['question'],
        'name': q_name
    })
    if 'similar_memories' in example:
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
            groups[-1]['examples'].insert(0, item['example'])
        

    for group in groups:
        meta = group['metadata']
        system_message = {
            'role': 'system',
            'content': f"{meta['participants']['q']} is interviewing you, {meta['participants']['a']}. It is the {meta['date']}.\n\nTo better answer the questions, some memories from your past writing will be retrieved if available, by the retrieve_memories function. It will be called automatically."
        }
        finetune_data = []

        for i, item in enumerate(group["examples"]):
            print(i)
            example_size = count_tokens(system_message)
            example, size = oaify_example(item, meta['participants'])
            examples = example
            example_size += size
            index = i
            if example_size < MAX_EMBEDDING_LENGTH:
                while example_size < MAX_EMBEDDING_LENGTH:
                    index = index + 1
                    print('index',index)
                    if index >= len(group["examples"]):
                        break
                    older_example = group["examples"][index]

                    example, size = oaify_example(older_example, meta['participants'])
                    if example_size + size < MAX_EMBEDDING_LENGTH:
                        examples = example + examples
                        example_size += size
            examples = [system_message] + examples
            finetune_data.append(examples)

    # Save the fine-tuned data to a new JSONL file
    with open('data/openai_finetune.jsonl', 'w') as f:
        for item in finetune_data:
            f.write(json.dumps({ "messages": item }))
            f.write('\n')
    return finetune_data

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