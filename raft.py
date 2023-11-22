# raft.py
import argparse
import os
from dotenv import load_dotenv
import openai
load_dotenv()
from src import files_helper, embeddings_helpers, substack_embeddings, generate_finetune, oai_finetune

# Load environment variables and set OpenAI API key

def main():
    parser = argparse.ArgumentParser(description='Run the raft command.')
    parser.add_argument('action', choices=['fetch', 'chunk', 'embed', 'ft:gen', 'ft:run', 'bench:setup'])
    parser.add_argument('name', help='The name of the blog to process.')
    parser.add_argument('--oai', help='Whether to generate the OAI format file.', default=False, action='store_true')
    args = parser.parse_args()

    if args.action == 'fetch':
        substack_embeddings.main(args.name)
    elif args.action == 'chunk':
        files_helper.chunker(args.name)
    elif args.action == 'embed':
        embeddings_helpers.store_grounding_embeddings(args.name)
    elif args.action == 'ft:gen':
        if args.oai:
            oai_finetune.create_openai_finetune_file(args.name)
        else:
            generate_finetune.generate_finetune(args.name)
    elif args.action == "ft:run":
        oai_finetune.run_oai_finetune(args.name)
    elif args.action == "bench:setup":
        generate_finetune.generate_benchmark(args.name)


    else:
        print(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main()