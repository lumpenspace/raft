"""
`raft interactive`: guided end-to-end session, in five phases.

1. gather -- collect documents (substack / RSS / URLs / PDFs / local
   files / LessWrong posts) and conversations (dumps, chat logs, tweets
   via ariadne, LessWrong comment threads), one source at a time,
   combined into one dataset.
2. prep   -- chunk + embed the corpus, then generate the finetune
   examples, each augmented with summaries of the target's relevant
   *earlier* writings.
3. train  -- pick the model and where it runs (the OpenAI finetuning
   API, a GPU pod via opbdh, or this machine's own accelerator); collect
   test questions while the job runs, previewing what retrieval puts in
   the persona's context.
4. eval   -- run the benchmark and the test questions against the
   finetuned model.
5. serve  -- talk to the persona, retrieval-augmented.

The session is resumable: it reads what already exists for the dataset
(see state.dataset_status) and suggests the next phase.
"""

import glob
import json
import tempfile
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List

import requests

from . import embeddings_helpers, files_helper, generate_finetune, hx, oai_finetune
from . import serve, sources, state, substack_embeddings
from .convo_structurer import import_conversation_file, import_text_source_file
from .hf_finetune import (
    default_local_target,
    is_openai_finetunable,
    load_opbdh,
    pick_model_interactively,
    run_hf_finetune,
)
from .interactive import ask, choose, confirm
from .memories import MemoryManager
from .project import DatasetLike, DatasetPaths, dataset_paths

PHASES = [
    ("gather", "collect documents and conversations"),
    ("prep", "chunk, embed, generate the finetune examples"),
    ("train", "pick a model and a venue, run the finetune"),
    ("eval", "try the benchmark and test questions on the result"),
    ("serve", "talk to the persona"),
]


def suggest_phase(status: dict) -> int:
    """The first phase that still has work to do, as a 0-based index."""
    if not status["corpus_docs"] and not status["transcripts"]:
        return 0
    if not status["openai_file"]:
        return 1
    if not status["model"]:
        return 2
    if not status["evaluated"]:
        return 3
    return 4


def phase_details(status: dict) -> List[str]:
    """One status line per phase, aligned with PHASES."""
    model = status["model"]
    return [
        f"{status['corpus_docs']} document(s), {status['transcripts']} conversation(s)",
        (
            f"{status['chunks']} chunk(s), "
            f"embedded: {'yes' if status['embedded'] else 'no'}, "
            f"examples: {'yes' if status['openai_file'] else 'no'}"
        ),
        f"model: {model}" if model else "no finetuned model yet",
        (
            f"{status['questions']} test question(s), "
            f"benchmark transcript: {'yes' if status['benchmark'] else 'no'}"
        ),
        f"ready ({model})" if model else "ready once a model is trained",
    ]


def render_phases(status: dict, suggested: int) -> None:
    """Print the phase map with done-marks (chrome, stderr)."""
    details = phase_details(status)
    c = hx.console()
    for i, (title, _) in enumerate(PHASES):
        # suggest_phase returns the first phase with work left, so
        # everything before it is done (and serve is never "done").
        done = i < suggested
        if c is None:
            mark = "x" if done else "-"
            print(f" {mark} {i + 1} {title:<8} {details[i]}", file=sys.stderr)
        else:
            mark = "[green]✓[/]" if done else "[dim]○[/]"
            c.print(
                f" {mark} [bold {hx.ACCENT}]{i + 1}[/] {title:<8} "
                f"[dim]{hx.esc(details[i])}[/]"
            )


def pick_dataset() -> tuple:
    """Choose an existing dataset to resume, or start a new one."""
    existing = state.list_datasets()
    if existing:
        default = 0 if len(existing) == 1 else len(existing)
        pick = choose("Dataset", existing + ["new dataset"], default=default)
        if pick < len(existing):
            name = existing[pick]
            target = state.load_meta(name).get("target", "")
            if not target:
                target = ask("Who is the target (the person to emulate)?", name)
                state.update_meta(name, target=target)
            return name, target
    target = ask("Who is the target (the person to emulate)?")
    while True:
        name = ask("Dataset name", target.replace(" ", "_").lower())
        # chroma collection names: 3+ chars of [a-zA-Z0-9._-], starting
        # and ending alphanumeric -- reject now, not at embed time.
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]+[a-zA-Z0-9]", name):
            break
        hx.warn("use 3+ characters: letters, digits, . _ - (alphanumeric ends)")
    state.update_meta(name, target=target)
    return name, target


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


def add_substack(name: str) -> None:
    """
    Fetch a whole substack (archive API) straight into the corpus --
    no data/{blog}-substack-com.jsonl side file, which pick_dataset
    would otherwise offer as a phantom dataset.
    """
    blog = ask("Substack subdomain (e.g. garymarcus)")
    hx.step(f"fetching {blog}.substack.com (polite delays between posts)")
    records = list(
        substack_embeddings.fetch_and_parse(f"https://{blog}.substack.com")
    )
    added = sources.append_corpus_records(name, records)
    hx.ok(f"{added} new document(s) from {blog}.substack.com")


SOURCE_KINDS = [
    "tweets (X / Bluesky)", "Substack", "blog / RSS / Atom feed",
    "web page / interview URL", "PDF files", "local text / JSONL files",
    "conversation files / chat logs", "LessWrong / EA Forum (posts + comments)",
]
TWEETS, LESSWRONG = 0, 7

# Sources that can feed both destinations at once, and how.
SPLIT_ROLES = {
    TWEETS: "both, split replies into conversations and other posts into grounding",
    LESSWRONG: "both: posts and quick takes as grounding, comment threads as conversations",
}


def plan_sources() -> list[dict]:
    """Collect all source choices and their roles before fetching anything."""
    plan = []
    hx.say("What sources do you have? Add as many as you like, then start importing.")
    hx.say("Conversations teach replies; grounding documents supply retrieval context. Grounding is optional.")
    while True:
        kind = choose("Available sources", SOURCE_KINDS + ["start importing"], default=len(SOURCE_KINDS) if plan else 0)
        if kind == len(SOURCE_KINDS):
            return plan
        roles = ["conversations", "grounding documents"]
        if kind in SPLIT_ROLES:
            roles.append(SPLIT_ROLES[kind])
        default = 2 if kind == LESSWRONG else 0 if kind in (TWEETS, 6) else 1
        role = choose(f"Use {SOURCE_KINDS[kind]} for", roles, default=default)
        plan.append({"kind": kind, "role": ["conversation", "corpus", "auto"][role]})
        hx.say(f"Added {SOURCE_KINDS[kind]}: {roles[role]}")


def import_planned_source(name: DatasetLike, target: str, item: dict) -> None:
    """Route a source only to its selected destination."""
    kind, role = item["kind"], item["role"]
    if kind == TWEETS:
        from .tweet_mode import run_tweet_mode
        run_tweet_mode(name, target, standalone=False, role=role)
        return
    if kind == LESSWRONG:
        from .lesswrong import run_lesswrong_source
        run_lesswrong_source(name, target, role=role)
        return
    if kind in (5, 6) and role == "conversation":
        gather_conversations(name, target)
        return
    # Stage fetched documents when extracting conversations, so interview text
    # never leaks into the destination grounding corpus.
    with tempfile.TemporaryDirectory(prefix="raft-source-") as directory:
        destination = DatasetPaths.legacy("source", directory) if role == "conversation" else name
        if kind == 1:
            add_substack(destination)
        elif kind == 2:
            sources.fetch_feed(destination, ask("Feed (or site) URL"),
                               fetch_pages=confirm("Fetch full pages for teaser entries?", default=True))
        elif kind == 3:
            sources.fetch_url(destination, ask("Page URL"))
        elif kind == 4:
            for path in collect_paths("PDF files"):
                sources.import_pdf(destination, path)
        else:
            for path in collect_paths("Text source files"):
                import_text_source_file(destination, path)
        if role == "conversation":
            from .convo_structurer import structure_raw_conversation, write_transcript
            corpus = dataset_paths(destination).corpus_path
            records = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()] if corpus.exists() else []
            for record in records:
                try:
                    transcript = structure_raw_conversation(record["content"], target)
                    write_transcript(name, transcript["participants"], record.get("date") or "unknown",
                                     record.get("link") or "", transcript["exchanges"])
                except ValueError as exc:
                    hx.warn(f"{record.get('title', 'document')}: {exc}")


def phase_gather(name: DatasetLike, target: str) -> None:
    """Plan multiple sources and their destinations, then import them."""
    plan = plan_sources()
    state.update_meta(name, source_plan=plan)
    for item in plan:
        hx.step(f"{SOURCE_KINDS[item['kind']]} -> {item['role']}")
        try:
            import_planned_source(name, target, item)
        except (requests.RequestException, ET.ParseError, ValueError, RuntimeError, OSError) as exc:
            hx.warn(f"source skipped: {exc}")
    status = state.dataset_status(name)
    hx.say(f"{status['corpus_docs']} grounding document(s), {status['transcripts']} conversation(s)")


def phase_prep(name: str) -> None:
    """Phase 2: chunk + embed the corpus, generate the finetune examples."""
    status = state.dataset_status(name)
    if not status["corpus_docs"] and not status["transcripts"]:
        hx.warn("nothing gathered yet -- add sources in the gather phase first")
        return

    if status["corpus_docs"]:
        if confirm("Chunk + embed the grounding corpus?"):
            files_helper.chunker(name)
            embeddings_helpers.store_grounding_embeddings(name)
    else:
        hx.say("Conversations only: skipping chunking and embedding; training examples will have no retrieved memories.")

    if not status["transcripts"]:
        hx.warn("no conversation examples yet -- add some in the gather phase")
        return
    if confirm(
        "Generate the finetune examples now?"
    ):
        thinking = confirm(
            "Format for a thinking model? (recall and a reasoning trace go in <think> blocks; "
            "Qwen3 and friends)",
            default=bool(state.load_meta(name).get("thinking")),
        )
        state.update_meta(name, thinking=thinking)
        generate_finetune.generate_finetune(name, thinking=thinking)
        oai_finetune.create_openai_finetune_file(name, thinking=thinking)


def collect_test_questions(name: str, when: str = "") -> None:
    """
    Gather test questions, previewing for each one which documents and
    tweets retrieval will put in the persona's context.
    """
    existing = state.test_questions(name)
    if existing:
        hx.say(f"{len(existing)} test question(s) stored so far")
    suffix = f" {when}" if when else ""
    hx.say(f"Add test questions{suffix} -- empty line to finish.")
    while True:
        question = ask("Test question", "")
        if not question:
            return
        state.add_test_question(name, question)
        serve.show_context(name, question)


def _wait_and_record(name: str, job_id: str) -> None:
    """Wait for a launched OpenAI job and record its model in the meta."""
    hx.step("waiting for the finetune to finish")
    model_id = oai_finetune.wait_oai_finetune(job_id)
    state.update_meta(name, pending_oai_job=None)
    if not model_id:
        hx.fail("the finetune job did not produce a model")
        return
    state.record_finetuned_model(name, model_id, "openai")
    hx.ok(f"finetuned model recorded: {model_id}")


def _resume_pending_job(name: str) -> bool:
    """
    Offer to re-attach to a previously launched OpenAI job (persisted in
    the meta so an interrupted session cannot orphan it). Returns True
    if the phase is handled.
    """
    pending = state.load_meta(name).get("pending_oai_job")
    if not pending:
        return False
    if not confirm(
        f"Job {pending['id']} ({pending.get('model', '?')}) was already "
        "launched -- wait for it instead of starting a new one?",
        default=True,
    ):
        state.update_meta(name, pending_oai_job=None)
        return False
    _wait_and_record(name, pending["id"])
    return True


def phase_train(name: str) -> None:
    """Phase 3: choose model + venue, run the finetune, gather questions."""
    status = state.dataset_status(name)
    if not status["openai_file"]:
        hx.warn("no finetune dataset yet -- run the prep phase first")
        return

    if _resume_pending_job(name):
        return

    venue = choose(
        "Where should the finetune run?",
        [
            "OpenAI finetuning API (hosted; gpt-4o-mini and friends)",
            "a GPU via opbdh (RunPod / Prime Intellect multi-cloud)",
            "this machine's accelerator via opbdh (Apple Silicon / CUDA)",
        ],
        default=0,
    )

    if venue == 0:
        while True:
            model = ask("OpenAI base model", "gpt-4o-mini-2024-07-18")
            if is_openai_finetunable(model):
                break
            hx.warn(
                f"{model} is not a known finetunable id "
                "(extend via RAFT_OAI_FINETUNABLE if it is new)"
            )
            if confirm("Use it anyway?", default=False):
                break
        if not confirm(
            f"Upload the dataset and start a paid finetune of {model} now?",
            default=False,
        ):
            return
        job_id = oai_finetune.launch_oai_finetune(name, model)
        state.update_meta(name, pending_oai_job={"id": job_id, "model": model})
        hx.ok(f"finetune job {job_id} launched (id saved in the dataset meta)")
        try:
            collect_test_questions(name, "while the job runs")
        except (EOFError, KeyboardInterrupt):
            hx.warn("question collection interrupted; the job keeps running")
        _wait_and_record(name, job_id)
    elif venue == 1:
        opbdh = load_opbdh()
        model = pick_model_interactively(opbdh)
        collect_test_questions(name, "before the pod spins up")
        if not confirm(
            f"Rent a GPU pod and start the {model} finetune now?", default=False
        ):
            return
        adapter = run_hf_finetune(name, model, interactive=True)
        if adapter:
            state.record_finetuned_model(name, adapter, "hf")
    else:
        opbdh = load_opbdh()
        target = default_local_target(opbdh)
        model = pick_model_interactively(opbdh)
        collect_test_questions(name, "before training starts")
        if not confirm(f"Start the {model} finetune on this machine ({target}) now?", default=True):
            return
        adapter = run_hf_finetune(name, model, opbdh_args=["--target", target], interactive=True)
        if adapter:
            state.record_finetuned_model(name, adapter, "hf")


def phase_eval(name: str) -> None:
    """Phase 4: benchmark files + test questions against the model."""
    model = state.finetuned_model(name)
    if not model:
        hx.warn("no finetuned model yet -- run the train phase first")
        return

    ran_something = False
    status = state.dataset_status(name)
    if status["benchmark"] and confirm(
        "Generate the benchmark files from the benchmark transcript?",
        default=not dataset_paths(name).benchmark_openai_path.exists(),
    ):
        thinking = bool(state.load_meta(name).get("thinking"))
        generate_finetune.generate_benchmark(name, thinking=thinking)
        oai_finetune.create_openai_finetune_file(name, "benchmark", thinking=thinking)
        ran_something = True

    questions = state.test_questions(name)
    if confirm("Add more test questions?", default=not questions):
        collect_test_questions(name)
        questions = state.test_questions(name)

    if not questions:
        hx.warn("no test questions to run")
    elif serve.is_local_adapter(name, model):
        hx.warn(
            "the finetuned model is a local adapter; raft cannot run it "
            "from here -- see the serving recipe in phase 5"
        )
    elif confirm(f"Run {len(questions)} test question(s) against {model}?"):
        manager = MemoryManager(name, {})
        thinking = bool(state.load_meta(name).get("thinking"))
        for question in questions:
            hx.step(question)
            serve.show_context(name, question)
            answer = manager.ask_question(question, model=model, thinking=thinking)
            print(f"\nQ: {question}\nA: {answer}\n")
        ran_something = True

    if ran_something:
        state.update_meta(
            name, evaluated_at=datetime.now(timezone.utc).date().isoformat()
        )


def run_interactive(dataset: DatasetLike | None = None) -> None:
    """Run the guided five-phase raft session."""
    hx.banner("build a persona dataset and finetune it")
    if dataset is None:
        name, target = pick_dataset()
    else:
        name = dataset
        target = state.load_meta(name).get("target") or ask("Who is the target (the person to emulate)?")
        state.update_meta(name, target=target)
    initial = state.dataset_status(name)
    if not initial["corpus_docs"] and not initial["transcripts"]:
        phase_gather(name, target)

    while True:
        status = state.dataset_status(name)
        suggested = suggest_phase(status)
        render_phases(status, suggested)
        pick = choose(
            "Phase",
            [f"{title} -- {desc}" for title, desc in PHASES] + ["quit"],
            default=suggested,
        )
        if pick == 0:
            phase_gather(name, target)
        elif pick == 1:
            phase_prep(name)
        elif pick == 2:
            phase_train(name)
        elif pick == 3:
            phase_eval(name)
        elif pick == 4:
            serve.run_serve(name, standalone=False)
        else:
            break

    hx.ok(f"all set. Come back with: raft interactive, or {dataset_paths(name).command('serve')}")
