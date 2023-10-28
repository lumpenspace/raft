# raft.py
import argparse
import os
from dotenv import load_dotenv
import openai

from src import ssjl, chunker, embeddings_helpers

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    parser = argparse.ArgumentParser(description='Run the raft command.')
    parser.add_argument('action', help='The action to perform.')
    parser.add_argument('name', help='The name of the blog to process.')
    args = parser.parse_args()

    if args.action == 'fetch':
        ssjl.main(args.name)
    elif args.action == 'chunk':
        chunker.main(args.name)
    elif args.action == 'embed':
        embeddings_helpers.store_grounding_embeddings(args.name)
    else:
        print(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main()