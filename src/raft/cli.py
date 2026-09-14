"""
This module contains the main CLI functionality for the RAFT project.
"""

import argparse
import sys
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

- init: Initialize a persona project in the current or specified directory.
- interactive: Guided session in five phases: gather, prep, train, eval, serve.
- tweets: Build a dataset from tweets via ariadne interactive.
- lesswrong: Import a LessWrong / EA Forum user: --user <handle> (--forum, --conversations,
  --min-karma, --role auto|corpus|conversation, --no-older-comments). Asks without --user.
- fetch: Fetch the blog from Substack and store it in the data directory.
- chunk: Chunk the blog into 4096 token pieces and store them in /data.
- embed: Create embeddings for the chunks and store them.
- ft:gen: Generate finetune files for the blog (--thinking for reasoning models).
- ft:run: Run the finetune job (OpenAI, or huggingface via opbdh).
- bench:setup: Setup the benchmark for the blog.
- ask: Ask a question about the blog content.
- serve: Chat with the finetuned persona, retrieval-augmented.
- comment: Have the persona comment on a post (--source <url|file>, or --web <port>).
  Needs RAFT_MLX_MODEL (an MLX-converted persona) or OPENAI_BASE_URL + --model.

ft:run routes by --model: OpenAI-finetunable ids go to the OpenAI API,
anything else (an org/name huggingface id) is trained via opbdh -- on a
GPU pod, or on this machine's accelerator with --target mps|cuda; extra
flags after the name configure the opbdh SFT recipe and GPU settings
(e.g. --method qlora --provider primeintellect, --epochs 1).
"""


cmds = [
    "init",
    "interactive",
    "tweets",
    "lesswrong",
    "fetch",
    "chunk",
    "embed",
    "ft:gen",
    "ft:run",
    "bench:setup",
    "ask",
    "serve",
    "comment",
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
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="ft:gen / bench:setup: format for a thinking model -- recall and a "
        "reasoning trace in <think> blocks (remembered for later --oai runs).",
    )
    parser.add_argument(
        "--recheck-traces",
        action="store_true",
        help="ft:gen: judge every reasoning trace in the generic file against its reply, "
        "rewrite the failures (no retrieval), then rebuild the chat file.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="comment: the post -- a URL (LessWrong, EA Forum, any page) or a text file.",
    )
    parser.add_argument(
        "--web",
        type=int,
        default=0,
        help="comment: serve a page on this port instead of commenting once.",
    )
    parser.add_argument(
        "--rewrite-traces",
        action="store_true",
        help="ft:gen: like --recheck-traces, but write every trace afresh first (a new writer model, say).",
    )
    parser.add_argument(
        "--user",
        default="",
        help="lesswrong: the forum username or profile slug to import (asks when omitted).",
    )
    parser.add_argument(
        "--forum",
        default="lesswrong",
        help="lesswrong: 'lesswrong' (default), 'eaforum', or a ForumMagnum base URL.",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=200,
        help="lesswrong: comment threads to import, newest first (0 = all; default 200).",
    )
    parser.add_argument(
        "--min-karma",
        type=int,
        default=None,
        help="lesswrong: skip comments scored below this.",
    )
    parser.add_argument(
        "--role",
        choices=["auto", "corpus", "conversation"],
        default="auto",
        help="lesswrong: what the forum feeds -- grounding (corpus), conversations, or both (auto).",
    )
    parser.add_argument(
        "--no-older-comments",
        action="store_true",
        help="lesswrong: leave the comments beyond the conversations out of the grounding documents.",
    )

    argv, passthrough = split_passthrough(sys.argv[1:], parser)
    args, extra = parser.parse_known_args(argv)
    extra = passthrough + extra

    if args.action not in ("ft:run",) and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    from .project import ProjectError, dataset_paths, find_project, initialize_project

    try:
        if args.action == "init":
            project, created = initialize_project(args.name or ".")
            print(f"{'Created' if created else 'Opened'} Raft project: {project.root}")
            return
        dataset = dataset_paths(args.name) if args.name else find_project()
    except ProjectError as exc:
        parser.error(str(exc))
    if dataset is None and args.action not in ("interactive", "tweets"):
        parser.error(f"the '{args.action}' action requires a dataset name or a Raft project (raft init)")

    if args.action == "interactive":
        from .flows import run_interactive

        run_interactive(dataset)
    elif args.action == "tweets":
        from .tweet_mode import run_tweet_mode

        run_tweet_mode(dataset or "")
    elif args.action == "lesswrong":
        from .lesswrong import run_lesswrong_cli, run_lesswrong_source

        try:
            if args.user:
                run_lesswrong_cli(
                    dataset,
                    args.user,
                    forum=args.forum,
                    max_conversations=args.conversations,
                    min_karma=args.min_karma,
                    role=args.role,
                    older_comments_as_grounding=not args.no_older_comments,
                )
            else:
                from .state import load_meta

                run_lesswrong_source(dataset, load_meta(dataset).get("target", ""), role=args.role)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.action == "fetch":
        if dataset.project:
            from .flows import add_substack
            add_substack(dataset)
        else:
            substack_embeddings.main(dataset.name)
    elif args.action == "chunk":
        files_helper.chunker(dataset)
    elif args.action == "embed":
        embeddings_helpers.store_grounding_embeddings(dataset)
    elif args.action == "ft:gen":
        thinking = _thinking_mode(dataset, args.thinking or args.recheck_traces or args.rewrite_traces)
        if args.recheck_traces or args.rewrite_traces:
            generate_finetune.recheck_traces(dataset, regenerate="all" if args.rewrite_traces else "recall")
            if not args.generic:
                oai_finetune.create_openai_finetune_file(dataset, thinking=True)
        elif args.oai:
            oai_finetune.create_openai_finetune_file(dataset, thinking=thinking)
        elif args.generic:
            generate_finetune.generate_finetune(dataset, thinking=thinking)
        else:
            generate_finetune.generate_finetune(dataset, thinking=thinking)
            oai_finetune.create_openai_finetune_file(dataset, thinking=thinking)
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
            model_id = oai_finetune.run_oai_finetune(dataset, model=model)
            if model_id:
                record_finetuned_model(dataset, model_id, "openai")
        else:
            adapter = run_hf_finetune(
                dataset,
                model,
                opbdh_args=extra,
                interactive=not args.no_interactive,
            )
            if adapter:
                record_finetuned_model(dataset, adapter, "hf")
    elif args.action == "bench:setup":
        thinking = _thinking_mode(dataset, args.thinking)
        if args.oai:
            oai_finetune.create_openai_finetune_file(dataset, "benchmark", thinking=thinking)
        elif args.generic:
            generate_finetune.generate_benchmark(dataset, thinking=thinking)
        else:
            generate_finetune.generate_benchmark(dataset, thinking=thinking)
            oai_finetune.create_openai_finetune_file(dataset, "benchmark", thinking=thinking)
    elif args.action == "serve":
        from .serve import run_serve

        run_serve(dataset, model=args.model)
    elif args.action == "comment":
        from .comment import run_comment

        run_comment(dataset, source=args.source, model=args.model, web=args.web)
    elif args.action == "ask":
        if args.question is None:
            print("Please provide a question using the --question argument.")
        else:
            memory_manager = memories.MemoryManager(
                dataset, {}
            )  # Empty metadata for now
            answer = memory_manager.ask_question(args.question)
            print(f"Answer: {answer}")
    else:
        print(f"Unknown action: {args.action}")


def split_passthrough(argv: list, parser: argparse.ArgumentParser) -> tuple:
    """
    Separate raft's own arguments from flags meant for the finetuning
    backend (`--target mps`, `--provider primeintellect`, `--epochs 2`).

    argparse would otherwise hand an unknown flag's value to the optional
    dataset-name positional, so inside a project `raft ft:run --target mps`
    became a dataset called "mps" with a bare --target. Anything after an
    unknown --flag belongs to that flag until the next --flag.
    """
    known = {option for action in parser._actions for option in action.option_strings}
    takes_value = {
        option for action in parser._actions for option in action.option_strings
        if action.nargs != 0 and not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
    }
    ours: list = []
    theirs: list = []
    i = 0
    while i < len(argv):
        token = argv[i]
        name = token.split("=", 1)[0]
        if token.startswith("--") and name not in known:
            theirs.append(token)
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                theirs.append(argv[i])
                i += 1
            continue
        ours.append(token)
        if token in takes_value and "=" not in token and i + 1 < len(argv):
            ours.append(argv[i + 1])
            i += 1
        i += 1
    return ours, theirs


def _thinking_mode(dataset, flag: bool) -> bool:
    """--thinking sticks to the dataset, so a later --oai pass keeps the format."""
    from . import state

    if flag:
        state.update_meta(dataset, thinking=True)
        return True
    return bool(state.load_meta(dataset).get("thinking"))


if __name__ == "__main__":
    main()
