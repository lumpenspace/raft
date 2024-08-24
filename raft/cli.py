# raft.py
import argparse
import os
from raft import (
    files_helper,
    embeddings_helpers,
    substack_embeddings,
    generate_finetune,
    oai_finetune,
    memories,
)


def actionDoc():
    return """
The following actions are available:

- fetch: Fetch the blog from Substack and store it in the data directory.
- chunk: Chunk the blog into 4096 token pieces and store them in the data directory.
- embed: Create embeddings for the chunks and store them.
- ft:gen: Generate finetune files for the blog.
- ft:run: Run the finetune job for the blog.
- bench:setup: Setup the benchmark for the blog.
- ask: Ask a question about the blog content.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run the raft command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=actionDoc(),
    )
    parser.add_argument(
        "action",
        help="The action to perform; see below for details.",
        choices=["fetch", "chunk", "embed", "ft:gen", "ft:run", "bench:setup", "ask"],
    )
    parser.add_help = actionDoc()
    parser.add_argument("name", help="The name of the blog to process.")
    parser.add_argument(
        "--oai",
        help="Only generate finetune or benchmark for openai (from existing generic file) .",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--generic",
        help="Only generate generic finetune or benchmark file.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--question",
        help="The question to ask about the blog content.",
        default=None,
    )
    parser.add_argument(
        "--no-useful-check",
        action="store_true",
        help="Skip usefulness check when summarizing memories",
    )
    args = parser.parse_args()

    if args.action == "fetch":
        substack_embeddings.main(args.name)
    elif args.action == "chunk":
        files_helper.chunker(args.name)
    elif args.action == "embed":
        embeddings_helpers.store_grounding_embeddings(args.name)
    elif args.action == "ft:gen":
        if args.oai:
            oai_finetune.create_openai_finetune_file(args.name, no_useful_check=args.no_useful_check)
        elif args.generic:
            generate_finetune.generate_finetune(args.name, no_useful_check=args.no_useful_check)
        else:
            generate_finetune.generate_finetune(args.name, no_useful_check=args.no_useful_check)
            oai_finetune.create_openai_finetune_file(args.name, no_useful_check=args.no_useful_check)
    elif args.action == "ft:run":
        oai_finetune.run_oai_finetune(args.name)
    elif args.action == "bench:setup":
        if args.oai:
            oai_finetune.create_openai_finetune_file(args.name, "benchmark", no_useful_check=args.no_useful_check)
        elif args.generic:
            generate_finetune.generate_finetune(args.name, no_useful_check=args.no_useful_check)
        else:
            generate_finetune.generate_benchmark(args.name, no_useful_check=args.no_useful_check)
            oai_finetune.create_openai_finetune_file(args.name, "benchmark", no_useful_check=args.no_useful_check)
    elif args.action == "ask":
        if args.question is None:
            print("Please provide a question using the --question argument.")
        else:
            memory_manager = memories.MemoryManager(
                args.name, {}
            )  # Empty metadata for now
            answer = memory_manager.ask_question(args.question, no_useful_check=args.no_useful_check)
            print(f"Answer: {answer}")
    else:
        print(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()