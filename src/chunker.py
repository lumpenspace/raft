import json
from typing import List, Dict
from openai import GPT3Encoder

MAX_EMBEDDING_LENGTH = 2048

def split_into_chunks(blog_posts: List[Dict], encoder: GPT3Encoder):
    for post in blog_posts:
        # Split the post content into chunks
        content_tokens = encoder.encode(post['content'])

        chunks = []
        chunk_start = 0
        while chunk_start < len(content_tokens):
            chunk_end = min(chunk_start + MAX_EMBEDDING_LENGTH, len(content_tokens))
            # Find the closest newline before the length limit
            while chunk_end > chunk_start and content_tokens[chunk_end] != encoder.encode('\n')[0]:
                chunk_end -= 1
            chunks.append(content_tokens[chunk_start:chunk_end])
            chunk_start = chunk_end

        # Add title, date and part to each chunk and store its embedding in the Chroma DB
        for j, chunk in enumerate(chunks):
            document = encoder.decode(chunk)
            metadata = {
                "title": post['title'],
                "url": post['url'],
                "part": j+1,
                "total_parts": len(chunks),
                "date": post['date']
            }
            yield document, metadata

def main(name):
    encoder = GPT3Encoder()
    sourcefile = f'{name}.json'
    outputfile = f'{name}_chunked.jsonl'

    with open(sourcefile, 'r') as f:
        blog_posts = json.load(f)

    with open(outputfile, 'w') as f:
        for document, metadata in split_into_chunks(blog_posts, encoder):
            f.write(json.dumps({"document": document, "metadata": metadata}) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Split blog posts into chunks.')
    parser.add_argument('name', help='The name of the blog to process.')
    args = parser.parse_args()
    main(args.name)