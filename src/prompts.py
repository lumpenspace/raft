from openai import OpenAI

client = OpenAI()

def get_interview_system_message(questioner:str, answerer:str, date:str):
    system_message = {
        'role': 'system',
        'content': f"{questioner} is interviewing you, {answerer}. It is the {date}.\n\nTo better answer the questions, some memories from your past writing will be retrieved if available, by the retrieve_memories function. It will be called automatically."
    }
    return system_message
    

def summarize_memory(memory, question:str, prev_answer, author:str):
    messages = [
        {
            "role": "system",
            "content": f"""You are helping {author} preparing for an interview.\n\
            Decide whether the quote from his blog, or extract from previous interview presented here is helpful in answering the question. \n\
            If it is, rephrase it from his perspective, in a way that would be helpful for answering, \
            in one or two sentences - type it directly, without intro. If it is not helpful, simply type 'skip'"""
        },
        {"role": "user", "content": f"Previous answer, for context: {prev_answer}\nQuestion: {question}\nMemory: {memory[0]}"}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content.strip()

def contextualise_memories_for_prompt(memories):
    memories_string = "\n".join([f"[memory]{memory[0]}" for memory in memories])
    if (len(memories_string) > 0):
        [{
            "role": "function",
            "name": "memory",
            "content": f"I wrote something relevant to this question in the past:\n{memories_string}"
        }]
    else:
        return []