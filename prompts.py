from openai import ChatCompletion

def summarize_memory(memory, question:str):
    messages = [
        {"role": "system", "content": "You are helping Paul Graham answer the questions from a young founders team. Decide whether the quote presented is helpful in answerting the question. If it is, rephrase it from his perspective, in a way that would be helpful for answeing, in one or two sentences - type it directly, without intro. Id it is not helpful, simply type 'skip'"},
        {"role": "user", "content": f"Question: {question}\nMemory: {memory[0]}"}
    ]
    response = ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)
    return response['choices'][0]['message']['content'].strip()

def contextualise_memories_for_prompt(memories):
    memories_string = "\n".join([f"[memory]{memory[0]}" for memory in memories])
    if (len(memories_string) > 0):
        [{ "role": "function", "name": "memory", "content": f"I wrote something relevant to this question in the past. To wit:\n{memories_string}" }]
    else:
        return []