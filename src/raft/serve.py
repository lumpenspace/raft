"""
`raft serve` / phase 5: talk to the finetuned persona.

Each turn shows what retrieval put in front of the model (which posts,
tweets and past exchanges), then the persona's retrieval-augmented
answer. Answers go to stdout so the session can be piped; all chrome
stays on stderr.

OpenAI finetunes are served directly. A huggingface LoRA adapter cannot
be run from here without the training stack, so for those we print a
short serving recipe instead.
"""

import os

from . import hx, state
from .interactive import ask, bail
from .memories import MemoryManager, preview_context

HF_SERVE_RECIPE = """\
That model is a local LoRA adapter; serve it with the peft stack, e.g.:

  from peft import AutoPeftModelForCausalLM
  from transformers import AutoTokenizer

  model = AutoPeftModelForCausalLM.from_pretrained("{adapter}")
  tokenizer = AutoTokenizer.from_pretrained(model.peft_config["default"].base_model_name_or_path)

or merge it and point vLLM / ollama at the result. Retrieval context for
each question is what `raft serve` shows above the answers -- the same
preview is available programmatically via raft.memories.preview_context."""


def is_local_adapter(name: str, model: str) -> bool:
    """
    Whether a model is a local LoRA adapter rather than a hosted id.

    The backend recorded at training time is authoritative; the path
    heuristic only covers models the user typed in by hand.
    """
    backend = state.model_backend(name, model)
    if backend:
        return backend == "hf"
    return os.path.exists(model) or (
        "/" in model and not model.startswith("ft:")
    )


def show_context(name: str, question: str) -> None:
    """Print the retrieval preview for a question (chrome, stderr)."""
    rows = preview_context(name, question)
    if not rows:
        hx.say("(nothing embedded yet -- no retrieval context)")
        return
    hx.say("in context:")
    for row in rows:
        date = f" ({row['date']})" if row["date"] else ""
        hx.say(f"  - {row['title']}{date}: {row['snippet']}")


def run_serve(name: str, model: str = "", standalone: bool = True) -> None:
    """
    Chat with the finetuned persona, retrieval-augmented.

    Args:
        name (str): Dataset name.
        model (str): Model to serve; defaults to the last finetuned
            model recorded for the dataset.
        standalone (bool): Print the banner (False inside `raft
            interactive`).
    """
    if standalone:
        hx.banner(f"talk to {name}")

    model = model or state.finetuned_model(name)
    if not model:
        model = ask("Model to serve (an ft:... id; empty to abort)", "")
        if not model:
            bail(f"no finetuned model recorded for {name} -- run `raft interactive`")

    if is_local_adapter(name, model):
        hx.say(HF_SERVE_RECIPE.format(adapter=model))
        return

    target = state.load_meta(name).get("target", name)
    manager = MemoryManager(name, {})
    hx.say(f"chatting with {target} ({model}); empty line to quit")

    while True:
        question = ask(f"To {target}", "").strip()
        if not question:
            return
        show_context(name, question)
        answer = manager.ask_question(question, model=model)
        print(f"\n{answer}\n")
