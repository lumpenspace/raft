import json
from typing import List, Dict, Tuple
import tiktoken
import pandas as pd    


MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def map_line (line: List[str]) -> Tuple[str, str, int]:
    encoded = encoding.encode(line)
    if len(encoded) < MAX_EMBEDDING_LENGTH:
        return (line, encoded, len(encoded))
    else:
        return (line[:MAX_EMBEDDING_LENGTH], encoded[:MAX_EMBEDDING_LENGTH], MAX_EMBEDDING_LENGTH)

def split_into_chunks(blog_posts: List[Dict]):
    for post in blog_posts:
        print(f"Splitting {post['title']}")
        lines = map(map_line, post['content'].split('\n'))

        # Calculate total parts
        total_parts = 0
        temp_chunk_length = 0
        for _, _, length in lines:
            if temp_chunk_length + length > MAX_EMBEDDING_LENGTH:
                total_parts += 1
                temp_chunk_length = length
            else:
                temp_chunk_length += length
        total_parts += 1  # for the last chunk

        chunk = []
        chunk_length = 0
        part = 1

        # Reset lines iterator
        lines = map(map_line, post['content'].split('\n'))

        for line, encoded, length in lines:
            if chunk_length + length > MAX_EMBEDDING_LENGTH:
                metadata = {
                    "title": post['title'],
                    "url": post['link'],
                    "date": post['date'].to_pydatetime().isoformat(),
                    "total_parts": total_parts,
                    "part": part
                }
                yield metadata, '\n'.join(chunk)
                chunk = []
                chunk_length = 0
                part += 1
            chunk.append(line)
            chunk_length += length

        if chunk:
            metadata = {
                "title": post['title'],
                "url": post['link'],
                "date": post['date'].to_pydatetime().isoformat(),
                "total_parts": total_parts,
                "part": part
            }
            yield metadata, '\n'.join(chunk)
            
def main(name):

    sourcefile = f'data/{name}.jsonl'
    outputfile = f'data/{name}_chunked.jsonl'


    blog_posts = pd.read_json(path_or_buf=sourcefile, lines=True).to_records()
    print(f"Splitting {len(blog_posts)} blog posts into chunks")

    with open(outputfile, 'w') as f:
        for item in split_into_chunks(blog_posts):
            f.write(json.dumps(item) + '\n')