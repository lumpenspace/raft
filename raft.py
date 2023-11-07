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
    parser.add_argument('action', choices=['fetch', 'chunk', 'embed', 'ft:gen'])
    parser.add_argument('name', help='The name of the blog to process.')
    parser.add_argument('--oai', help='Whether to generate the OAI format file.', default=False, action='store_true')
    args = parser.parse_args()

    if args.action == 'fetch':
        ssjl.main(args.name)
    elif args.action == 'chunk':
        chunker.main(args.name)
    elif args.action == 'embed':
        embeddings_helpers.store_grounding_embeddings(args.name)
    elif args.action == 'ft:gen':
        if args.oai:
            from src import oai_finetune

            oai_finetune.create_openai_finetune_file(args.name)
        else:
            from src import generate_finetune

            generate_finetune.generate_finetune(args.name)
    else:
        print(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main()