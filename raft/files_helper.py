from typing import Dict, Tuple, Any, Generator
import json
import tiktoken
import pandas as pd
from pandas import DataFrame
from datetime import datetime


MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def begin_json_file(name: str):
    # Open the output file and write the opening bracket for the JSON array
    write_to_file(name, "[\n", "w")


def end_json_file(name: str):
    # Open the output file in append mode and
    # write the closing bracket for the JSON array
    write_to_file(name, "\n]", "a")


def write_to_file(name: str, data: str, mode: str = "a"):
    with open(f"data/{name}.json", mode) as f:
        f.write(data)


def write_context_to_file(filename: str, context: Dict[str, Any], suffix: int, j: int):
    # Open the output file in append mode and write the context
    with open(f"data/{filename}", "a") as f:
        # If it's not the first item, prepend a comma
        if suffix > 1 or j > 0:
            f.write(",\n")
        json.dump(context, f, indent=4)


def map_line(line: str) -> Tuple[str, str, int]:
    encoded = encoding.encode(line)
    if len(encoded) < MAX_EMBEDDING_LENGTH:
        return (line, encoding.decode(encoded), len(encoded))
    else:
        truncated_encoded = encoded[:MAX_EMBEDDING_LENGTH]
        return (
            line[:MAX_EMBEDDING_LENGTH],
            encoding.decode(truncated_encoded),
            MAX_EMBEDDING_LENGTH,
        )


def split_into_chunks(
    blog_posts: DataFrame,
) -> Generator[Tuple[Dict[str, Any], str], None, None]:
    for _, post in blog_posts.iterrows():
        print(f"Splitting {post['title']}")
        lines = list(map(map_line, post["content"].split("\n")))

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

        for line, _, length in lines:
            if chunk_length + length > MAX_EMBEDDING_LENGTH:
                metadata = {
                    "title": post["title"],
                    "url": post["link"],
                    "date": post["date"] if isinstance(post["date"], str) else post["date"].isoformat(),
                    "total_parts": total_parts,
                    "part": part,
                }
                yield metadata, "\n".join(chunk)
                chunk = []
                chunk_length = 0
                part += 1
            chunk.append(line)
            chunk_length += length

        if chunk:
            metadata = {
                "title": post["title"],
                "url": post["link"],
                "date": post["date"] if isinstance(post["date"], str) else post["date"].isoformat(),
                "total_parts": total_parts,
                "part": part,
            }
            yield metadata, "\n".join(chunk)


def chunker(name: str):
    sourcefile = f"data/{name}.jsonl"
    outputfile = f"data/{name}_chunked.jsonl"

    try:
        # Read the JSONL file line by line
        with open(sourcefile, 'r') as f:
            data = [json.loads(line) for line in f]
        
        blog_posts: DataFrame = pd.DataFrame(data)
        print(f"Splitting {len(blog_posts)} blog posts into chunks")

        with open(outputfile, "w") as f:
            for item in split_into_chunks(blog_posts):
                f.write(json.dumps(item) + "\n")
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Error occurred at line {e.lineno}")
        with open(sourcefile, 'r') as f:
            problematic_line = f.readlines()[e.lineno - 1]
        print(f"Problematic line: {problematic_line}")
    
    except Exception as e:
        print(f"An error occurred: {e}")