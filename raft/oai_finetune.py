from openai import OpenAI
import time
import json
import tiktoken
from .prompt_manager import PromptManager
from .files_helper import begin_json_file, end_json_file, write_context_to_file

prompt_manager = PromptManager()  # Remove the client argument


MAX_FINETUNE_LENGTH = 4096
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
        'name': q_name.replace(' ', '')
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
        'name': a_name.replace(' ', '')
    })
    return result, len(encoding.encode(json.dumps(result)));
    
def create_finetune_job(name:str, file, model:str):
    print(f"creating finetune job for: {model}")
    job = client.fine_tuning.jobs.create(training_file=file.id, model=model, suffix=name)
    print(f'Fine tune started for job: {job.id} with model: {model}')
    return job

def run_oai_finetune(name:str):
    models = ["gpt-3.5-turbo"]
    status_dict = {model: "" for model in models}
    filename = f'data/{name}_finetune_openai.jsonl'

    print(f'uploading file: {filename}')
    # get file object
    source_file = open(file=filename, mode='rb')

    file = client.files.create(file=source_file, purpose="fine-tune")
    for model in models:
        job = create_finetune_job(name, file, model)

        while True:
            job_update = client.fine_tuning.jobs.retrieve(job.id)
            if job_update.status in ['succeeded', 'failed']:    
                print(f"Fine tune completed for model {model}. Model ID: {job_update.fine_tuned_model}")
                status_dict[model] = job_update.status
                break
            if (status_dict[model] != job_update.status):
                print(f"Fine tune status for model {model}:")
                print(job_update.status)
                status_dict[model] = job_update.status
            time.sleep(2)

def create_openai_finetune_file(name: str, type:str="finetune"):
    with open(f"data/{name}_{type}.json") as f:
        data = json.load(f) 

    # group the examples and reverse the order within each group
    groups = []
    for item in data:
        if 'metadata' in item:
            groups.append({ "metadata": item['metadata'], "examples": [] })
        elif 'example' in item:
            groups[-1]['examples'].insert(0, item['example'])
        
    finetune_data = []
    for group in groups:
        meta = group['metadata']
        system_message = prompt_manager.get_interview_system_message(questioner=meta['participants']['q'], answerer=meta['participants']['a'], date=meta['date'])

        group_data = []

        for i, item in enumerate(group["examples"]):
            print(i)
            example_size = count_tokens(system_message)
            example, size = oaify_example(item, meta['participants'])
            examples = example
            example_size += size
            index = i
            if example_size < MAX_FINETUNE_LENGTH:
                while example_size < MAX_FINETUNE_LENGTH:
                    index = index + 1
                    print('index',index)
                    if index >= len(group["examples"]):
                        break
                    older_example = group["examples"][index]

                    example, size = oaify_example(older_example, meta['participants'])
                    if example_size + size < MAX_FINETUNE_LENGTH:
                        examples = example + examples
                        example_size += size
            examples = [system_message] + examples
            group_data.append(examples)
        finetune_data = group_data + finetune_data

    # Save the fine-tuned data to a new JSONL file
    with open(f'data/{name}_{type}_openai.jsonl', 'w') as f:
        finetune_data.reverse()
        for item in finetune_data:
            f.write(json.dumps({ "messages": item }))
            f.write('\n')
    return finetune_data