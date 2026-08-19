"""
This module contains the chunker function, which splits blog posts into chunks
of a maximum length.
"""

import json
from typing import List, Dict, Tuple, Generator
import tiktoken
import pandas as pd

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def map_line(line: str) -> Tuple[str, str, int]:
    """
    Encode and potentially truncate a line of text.

    Args:
        line (str): The input line of text.

    Returns:
        Tuple[str, str, int]: A tuple containing:
            - the original or truncated line,
            - the decoded version, and
            - the length of the encoded line.
    """
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
    blog_posts: pd.DataFrame,
) -> Generator[Tuple[Dict[str, str], str], None, None]:
    """
    Split blog posts into chunks of a maximum length.

    Args:
        blog_posts (pd.DataFrame): DataFrame containing blog post data.

    Yields:
        Tuple[Dict[str, str], str]: A tuple containing metadata
            and the chunked content.
    """
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

        chunk: List[str] = []
        chunk_length = 0
        part = 1

        for line, _, length in lines:
            if chunk_length + length > MAX_EMBEDDING_LENGTH:
                metadata = {
                    "title": post["title"],
                    "url": post["link"],
                    "date": (
                        post["date"].isoformat()
                        if isinstance(post["date"], pd.Timestamp)
                        else post["date"]
                    ),
                    "total_parts": str(total_parts),
                    "part": str(part),
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
                "date": (
                    post["date"].isoformat()
                    if isinstance(post["date"], pd.Timestamp)
                    else post["date"]
                ),
                "total_parts": str(total_parts),
                "part": str(part),
            }
            yield metadata, "\n".join(chunk)


def chunker(name: str) -> None:
    """
    Process a JSONL file of blog posts, split them into chunks,
    and save the results.

    Args:
        name (str): The name of the file to process (without extension).
    """
    sourcefile = f"data/{name}.jsonl"
    outputfile = f"data/{name}_chunked.jsonl"

    try:
        # Read the JSONL file line by line
        with open(sourcefile, "r") as f:
            data = [json.loads(line) for line in f]

        blog_posts: pd.DataFrame = pd.DataFrame(data)
        print(f"Splitting {len(blog_posts)} blog posts into chunks")

        with open(outputfile, "w") as f:
            for item in split_into_chunks(blog_posts):
                f.write(json.dumps(item) + "\n")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Error occurred at line {e.lineno}")
        with open(sourcefile, "r") as f:
            problematic_line = f.readlines()[e.lineno - 1]
        print(f"Problematic line: {problematic_line}")

    except Exception as e:
        print(f"An error occurred: {e}")
