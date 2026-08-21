"""
This module contains the main CLI functionality for the RAFT project.
"""

import argparse
from raft import (
    files_helper,
    embeddings_helpers,
    substack_embeddings,
    generate_finetune,
    oai_finetune,
    memories,
)


def action_doc() -> str:
    """
    Return a string describing available actions for the CLI.

    Returns:
        str: Description of available actions.
    """
    return """
The following actions are available:

- interactive: Guided session in five phases: gather, prep, train, eval, serve.
- tweets: Build a dataset from tweets via ariadne interactive.
- fetch: Fetch the blog from Substack and store it in the data directory.
- chunk: Chunk the blog into 4096 token pieces and store them in /data.
- embed: Create embeddings for the chunks and store them.
- ft:gen: Generate finetune files for the blog.
- ft:run: Run the finetune job (OpenAI, or huggingface via opbdh).
- bench:setup: Setup the benchmark for the blog.
- ask: Ask a question about the blog content.
- serve: Chat with the finetuned persona, retrieval-augmented.

ft:run routes by --model: OpenAI-finetunable ids go to the OpenAI API,
anything else (an org/name huggingface id) is trained on a GPU pod via
opbdh; unknown flags after the name are forwarded verbatim to
`opbdh launch` (e.g. --vram-gb 48 --max-spend 5).
"""


cmds = [
    "interactive",
    "tweets",
    "fetch",
    "chunk",
    "embed",
    "ft:gen",
    "ft:run",
    "bench:setup",
    "ask",
    "serve",
]


def main() -> None:
    """
    Main function to handle CLI arguments and execute corresponding actions.
    """
    parser = argparse.ArgumentParser(
        description="Run the raft command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=action_doc(),
    )
    parser.add_argument(
        "action",
        help="The action to perform; see below for details.",
        choices=cmds,
    )

    parser.add_argument(
        "name",
        nargs="?",
        default="",
        help="The name of the dataset to process.",
    )
    parser.add_argument(
        "--oai",
        help="Only generate finetune or benchmark for openai \
            (from existing generic file).",
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
    parser.add_argument(
        "--model",
        help="Model to finetune (ft:run): an OpenAI id, or a \
            huggingface org/name id trained via opbdh.",
        default="",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="ft:run: never prompt; rely on flags and opbdh config.",
    )

    args, extra = parser.parse_known_args()

    if args.action not in ("ft:run",) and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    needs_name = args.action not in ("interactive", "tweets")
    if needs_name and not args.name:
        parser.error(f"the '{args.action}' action requires a dataset name")

    if args.action == "interactive":
        from .flows import run_interactive

        run_interactive()
    elif args.action == "tweets":
        from .tweet_mode import run_tweet_mode

        run_tweet_mode(args.name)
    elif args.action == "fetch":
        substack_embeddings.main(args.name)
    elif args.action == "chunk":
        files_helper.chunker(args.name)
    elif args.action == "embed":
        embeddings_helpers.store_grounding_embeddings(args.name)
    elif args.action == "ft:gen":
        if args.oai:
            oai_finetune.create_openai_finetune_file(args.name)
        elif args.generic:
            generate_finetune.generate_finetune(args.name)
        else:
            generate_finetune.generate_finetune(args.name)
            oai_finetune.create_openai_finetune_file(args.name)
    elif args.action == "ft:run":
        from .hf_finetune import is_openai_finetunable, run_hf_finetune
        from .interactive import ask

        model = args.model
        if not model and not args.no_interactive:
            model = ask(
                "Model to finetune (OpenAI id or huggingface org/name)",
                "gpt-4o-mini-2024-07-18",
            )
        from .state import record_finetuned_model

        if is_openai_finetunable(model):
            if extra:
                parser.error(
                    "opbdh flags only apply to huggingface models: "
                    f"{' '.join(extra)}"
                )
            model_id = oai_finetune.run_oai_finetune(args.name, model=model)
            if model_id:
                record_finetuned_model(args.name, model_id, "openai")
        else:
            adapter = run_hf_finetune(
                args.name,
                model,
                opbdh_args=extra,
                interactive=not args.no_interactive,
            )
            record_finetuned_model(args.name, adapter, "hf")
    elif args.action == "bench:setup":
        if args.oai:
            oai_finetune.create_openai_finetune_file(args.name, "benchmark")
        elif args.generic:
            generate_finetune.generate_finetune(args.name)
        else:
            generate_finetune.generate_benchmark(args.name)
            oai_finetune.create_openai_finetune_file(args.name, "benchmark")
    elif args.action == "serve":
        from .serve import run_serve

        run_serve(args.name, model=args.model)
    elif args.action == "ask":
        if args.question is None:
            print("Please provide a question using the --question argument.")
        else:
            memory_manager = memories.MemoryManager(
                args.name, {}
            )  # Empty metadata for now
            answer = memory_manager.ask_question(args.question)
            print(f"Answer: {answer}")
    else:
        print(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()
