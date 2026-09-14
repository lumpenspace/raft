"""
This module contains the main CLI functionality for the RAFT project.
"""

import argparse
import sys
from raft import (
    files_helper,
    embeddings_helpers,
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
- fetch <source>: Import one source into the project. Sources:
    lesswrong  a LessWrong / EA Forum user: --user <handle> (--forum, --min-karma,
               --no-older-comments)
    tweets     X / Bluesky through ariadne: --user <handle> (a dotted handle is Bluesky)
               or --archive <X export>; --network x|bluesky|both
    substack   a whole publication as grounding: --blog <subdomain>
    rss        a blog, RSS or Atom feed as grounding: --url <feed or site> (--no-full-pages)
    url        one web page as grounding: --url <page>
    pdf        PDF files as grounding: --file <path> (repeatable)
  Shared: --since / --until YYYY-MM-DD keep only what was written in the window;
  --limit N keeps the newest N (conversations for lesswrong and tweets, documents
  otherwise; lesswrong defaults to 200); --role auto|corpus|conversation for the
  two sources that feed both. Each source asks for what its flags leave out.
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

FETCH_SOURCES = ["lesswrong", "tweets", "substack", "rss", "url", "pdf"]

# `raft lesswrong` and `raft tweets` (3.0) became `raft fetch <source>` in 3.1.
LEGACY_FETCH_ACTIONS = ("lesswrong", "tweets")


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
        "rest",
        nargs="*",
        metavar="source|name",
        help="fetch: the source (lesswrong, tweets, substack, rss, url, pdf). "
        "Outside a project, the dataset name comes after it.",
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
        help="fetch lesswrong / tweets: the handle to import (a forum username or slug; an X handle, "
        "or a dotted Bluesky one). Asks when omitted.",
    )
    parser.add_argument(
        "--archive",
        default="",
        help="fetch tweets: an X archive export (zip or folder); with --user, only that handle's tweets.",
    )
    parser.add_argument(
        "--network",
        choices=["x", "bluesky", "both"],
        default="",
        help="fetch tweets: which network --user is on (guessed from the handle when omitted).",
    )
    parser.add_argument(
        "--since",
        default="",
        help="fetch: only what was written on or after this day (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--until",
        default="",
        help="fetch: only what was written on or before this day (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="fetch: keep the newest N -- conversations for lesswrong and tweets, documents for the rest "
        "(lesswrong defaults to 200; 0 = all).",
    )
    parser.add_argument(
        "--forum",
        default="lesswrong",
        help="fetch lesswrong: 'lesswrong' (default), 'eaforum', or a ForumMagnum base URL.",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=None,
        help="fetch lesswrong: the 3.0 name of --limit.",
    )
    parser.add_argument(
        "--min-karma",
        type=int,
        default=None,
        help="fetch lesswrong: skip comments scored below this.",
    )
    parser.add_argument(
        "--role",
        choices=["auto", "corpus", "conversation"],
        default="auto",
        help="fetch lesswrong / tweets: what the source feeds -- grounding (corpus), conversations, or both (auto).",
    )
    parser.add_argument(
        "--no-older-comments",
        action="store_true",
        help="fetch lesswrong: leave the comments beyond the conversations out of the grounding documents.",
    )
    parser.add_argument(
        "--blog",
        default="",
        help="fetch substack: the publication's subdomain (garymarcus for garymarcus.substack.com).",
    )
    parser.add_argument(
        "--url",
        default="",
        help="fetch rss / url: the feed (or site) URL, or the page URL.",
    )
    parser.add_argument(
        "--no-full-pages",
        action="store_true",
        help="fetch rss: keep teaser entries as they are instead of fetching their pages.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="fetch pdf: a PDF to import (repeat for several).",
    )

    argv = list(sys.argv[1:])
    if argv and argv[0] in LEGACY_FETCH_ACTIONS:
        from . import hx

        hx.warn(f"`raft {argv[0]}` is now `raft fetch {argv[0]}`")
        argv.insert(0, "fetch")
    argv, passthrough = split_passthrough(argv, parser)
    args, extra = parser.parse_known_args(argv)
    extra = passthrough + extra

    if args.action not in ("ft:run",) and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    positionals = list(args.rest)
    source = ""
    if args.action == "fetch":
        if not positionals or positionals[0] not in FETCH_SOURCES:
            parser.error(f"fetch needs a source: {', '.join(FETCH_SOURCES)}")
        source = positionals.pop(0)
        for flag in ("since", "until"):
            raw = getattr(args, flag)
            if raw and not _iso_day(raw):
                parser.error(f"--{flag} must be a day, YYYY-MM-DD, not {raw!r}")
            setattr(args, flag, _iso_day(raw))
        if args.since and args.until and args.since > args.until:
            parser.error("--since is later than --until")
        if args.limit is None:
            args.limit = args.conversations
    if len(positionals) > 1:
        parser.error(f"unrecognized arguments: {' '.join(positionals[1:])}")
    name = positionals[0] if positionals else ""

    from .project import ProjectError, dataset_paths, find_project, initialize_project

    try:
        if args.action == "init":
            project, created = initialize_project(name or ".")
            print(f"{'Created' if created else 'Opened'} Raft project: {project.root}")
            return
        dataset = dataset_paths(name) if name else find_project()
    except ProjectError as exc:
        parser.error(str(exc))
    if dataset is None and not (args.action == "interactive" or source == "tweets"):
        parser.error(f"the '{args.action}' action requires a dataset name or a Raft project (raft init)")

    if args.action == "interactive":
        from .flows import run_interactive

        run_interactive(dataset)
    elif args.action == "fetch":
        run_fetch(parser, dataset, source, args)
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


def run_fetch(parser: argparse.ArgumentParser, dataset, source: str, args: argparse.Namespace) -> None:
    """
    `raft fetch <source>`: import one source into the dataset.

    lesswrong and tweets feed conversations and grounding (--role); the
    others feed grounding only. --since / --until / --limit apply to all
    of them (url and pdf have no dates or lists to cut). Every source
    asks for what its flags leave out, so `raft fetch substack` works
    like the wizard's step.
    """
    import xml.etree.ElementTree as ET

    import requests

    from . import hx, sources
    from .interactive import ask

    try:
        if source == "lesswrong":
            from .lesswrong import run_lesswrong_cli, run_lesswrong_source

            if args.user:
                run_lesswrong_cli(
                    dataset,
                    args.user,
                    forum=args.forum,
                    max_conversations=200 if args.limit is None else args.limit,
                    min_karma=args.min_karma,
                    role=args.role,
                    older_comments_as_grounding=not args.no_older_comments,
                    since=args.since,
                    until=args.until,
                )
            else:
                from .state import load_meta

                run_lesswrong_source(dataset, load_meta(dataset).get("target", ""), role=args.role)
        elif source == "tweets":
            from .tweet_mode import run_tweet_mode

            run_tweet_mode(
                dataset or "", role=args.role, user=args.user, archive=args.archive, network=args.network,
                since=args.since, until=args.until, limit=args.limit or None,
            )
        elif source == "substack":
            from .flows import add_substack

            add_substack(dataset, args.blog, since=args.since, until=args.until, limit=args.limit or None)
        elif source == "rss":
            url = args.url or ask("Feed (or site) URL")
            added = sources.fetch_feed(
                dataset, url, fetch_pages=not args.no_full_pages,
                since=args.since, until=args.until, limit=args.limit or None,
            )
            hx.ok(f"{added} new document(s) from {url}")
        elif source == "url":
            url = args.url or ask("Page URL")
            added = sources.fetch_url(dataset, url)
            hx.ok(f"{added} new document(s) from {url}")
        elif source == "pdf":
            from .flows import collect_paths

            paths = args.file or collect_paths("PDF files")
            added = sum(sources.import_pdf(dataset, path) for path in paths)
            hx.ok(f"{added} new document(s) from {len(paths)} PDF(s)")
    except (requests.RequestException, ET.ParseError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


def _iso_day(value: str) -> str:
    """A --since / --until value as YYYY-MM-DD, or "" when it is not a day."""
    from .sources import iso_date

    return iso_date(value)


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
