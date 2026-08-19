"""
`raft interactive`: guided end-to-end session.

Collects text sources (substack / tweets via ariadne / local files) and
conversation examples (structured or not; unstructured ones are
converted with an LLM), builds the grounding db and the finetune
dataset, then routes the finetune to OpenAI or -- via opbdh -- to a
huggingface model on a GPU pod.
"""

import glob
import os
from typing import List

from . import embeddings_helpers, files_helper, generate_finetune, hx, oai_finetune
from . import substack_embeddings
from .convo_structurer import import_conversation_file, import_text_source_file
from .hf_finetune import is_openai_finetunable, run_hf_finetune
from .interactive import ask, choose, confirm


def collect_paths(prompt: str) -> List[str]:
    """Ask for file paths (globs allowed) until the user is done."""
    paths: List[str] = []
    hx.say(f"{prompt} (one per line, globs ok; empty line to finish)")
    while True:
        raw = input("> ").strip()
        if not raw:
            return paths
        expanded = sorted(glob.glob(os.path.expanduser(raw)))
        if not expanded:
            hx.warn("no files match")
        paths.extend(expanded)


def gather_text_sources(name: str, target: str) -> bool:
    """Collect grounding text sources. Returns True if any were added."""
    added = False
    while True:
        kind = choose(
            "Add a text source for the grounding corpus?",
            [
                "substack blog (fetched automatically)",
                "tweets (via ariadne interactive)",
                "local files (structured jsonl or raw text)",
                "done",
            ],
            default=3,
        )
        if kind == 0:
            blog = ask("Substack subdomain (e.g. garymarcus)")
            substack_embeddings.main(blog)
            fetched = f"data/{blog}-substack-com.jsonl"
            if os.path.exists(fetched) and fetched != f"data/{name}.jsonl":
                import_text_source_file(name, fetched)
            added = True
        elif kind == 1:
            from .tweet_mode import run_tweet_mode

            run_tweet_mode(name, target, standalone=False)
            added = True
        elif kind == 2:
            for path in collect_paths("Text source files"):
                n = import_text_source_file(name, path)
                hx.ok(f"{path}: {n} document(s) added")
                added = True
        else:
            return added


def gather_conversations(name: str, target: str) -> bool:
    """Collect conversation examples. Returns True if any were added."""
    added = False
    paths = collect_paths(
        "Conversation example files (transcripts, chat logs, raw text --\n"
        "unstructured files are converted with an LLM)"
    )
    for path in paths:
        try:
            written = import_conversation_file(name, path, target)
            hx.ok(f"{path} -> {written}")
            added = True
        except ValueError as e:
            hx.warn(f"skipped: {e}")
    return added


def run_interactive() -> None:
    """Run the guided end-to-end raft session."""
    hx.banner("build a persona dataset and finetune it")
    target = ask("Who is the target (the person to emulate)?")
    name = ask("Dataset name", target.replace(" ", "_").lower())

    gather_text_sources(name, target)
    gather_conversations(name, target)

    has_corpus = os.path.exists(f"data/{name}.jsonl")
    if has_corpus and confirm("Chunk + embed the grounding corpus now?"):
        files_helper.chunker(name)
        embeddings_helpers.store_grounding_embeddings(name)

    has_transcripts = os.path.exists(f"data/{name}_transcript_1.json")
    if has_transcripts and confirm("Generate the finetune dataset now?"):
        generate_finetune.generate_finetune(name)
        oai_finetune.create_openai_finetune_file(name)

    if os.path.exists(f"data/{name}_finetune_openai.jsonl") and confirm(
        "Run the finetune now?", default=False
    ):
        model = ask("Model to finetune", "gpt-4o-mini-2024-07-18")
        if is_openai_finetunable(model):
            oai_finetune.run_oai_finetune(name, model=model)
        else:
            run_hf_finetune(name, model, interactive=True)

    hx.ok(f"all set. Try: raft ask {name} --question '...'")
