"""Hugging Face fine-tuning through OPBDH's native SFT facility."""

from dataclasses import asdict
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

from . import hx
from .interactive import ask, bail, choose, confirm
from .project import DatasetLike, dataset_paths

OPBDH_INSTALL_HINT = (
    "opbdh is not installed. Install it with:\n"
    "  pip install -U 'opbdh[ft]>=1.10.0'\n"
    "then run `opbdh config wizard` to set up your provider credentials."
)

# Model-name prefixes OpenAI accepts for finetuning; anything else is
# routed to huggingface via opbdh. Extend via RAFT_OAI_FINETUNABLE
# (comma-separated prefixes).
OPENAI_FINETUNABLE_PREFIXES = (
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o-2024",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1-2025",
    "davinci-002",
    "babbage-002",
)

def is_openai_finetunable(model: str) -> bool:
    """Check whether a model id can be finetuned on the OpenAI API."""
    extra = tuple(
        p.strip()
        for p in os.environ.get("RAFT_OAI_FINETUNABLE", "").split(",")
        if p.strip()
    )
    return model.startswith(OPENAI_FINETUNABLE_PREFIXES + extra)


def load_opbdh():
    """Import opbdh, or exit with an install hint."""
    try:
        import opbdh
    except ImportError:
        bail(OPBDH_INSTALL_HINT)
    try:
        from opbdh.finetune import require_finetune_extra
        require_finetune_extra()
    except (ImportError, RuntimeError):
        bail(OPBDH_INSTALL_HINT)
    return opbdh


def parse_opbdh_flags(flags: List[str]) -> Dict[str, Any]:
    """
    Turn passthrough CLI flags into opbdh keyword overrides.

    `--vram-gb 48 --max-spend 5` becomes `{"vram_gb": 48, "max_spend": 5.0}`;
    a flag with no value is treated as a boolean switch.
    """
    overrides: Dict[str, Any] = {}
    index = 0
    while index < len(flags):
        flag = flags[index]
        if not flag.startswith("--"):
            bail(f"unexpected argument for opbdh: {flag}")
        key = flag[2:].replace("-", "_")
        if "=" in key:
            key, raw = key.split("=", 1)
            index += 1
        elif index + 1 < len(flags) and not flags[index + 1].startswith("--"):
            raw = flags[index + 1]
            index += 2
        else:
            overrides[key] = True
            index += 1
            continue
        overrides[key] = coerce_flag_value(raw)
    return overrides


def coerce_flag_value(raw: str) -> Any:
    """Coerce a CLI flag string to int/float/bool where it plainly is one."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


# Training flags belong to OPBDH's SFT recipe, not its pod configuration.
TRAINING_FIELDS = {
    "epochs", "learning_rate", "max_length", "per_device_batch_size",
    "gradient_accumulation_steps", "lora_r", "lora_alpha", "lora_dropout",
    "packing", "seed", "trust_remote_code", "chat_template",
}
FLAG_ALIASES = {"batch_size": "per_device_batch_size", "gradient_accumulation": "gradient_accumulation_steps"}


def prepare_finetune(dataset: DatasetLike, model: str, options: Dict[str, Any]):
    """Import RAFT examples into a resumable native OPBDH recipe and job."""
    opbdh = load_opbdh()
    from opbdh import finetune as ft
    paths = dataset_paths(dataset)
    source = paths.finetune_openai_path
    if not source.exists():
        bail(f"{source} not found -- run `{paths.command('ft:gen')}` first")
    options = {FLAG_ALIASES.get(k, k): v for k, v in options.items()}
    training = {key: options.pop(key) for key in list(options) if key in TRAINING_FIELDS}
    method = options.pop("method", None)
    recipe = options.pop("recipe", None)
    if method is not None and method not in ("lora", "qlora"):
        raise ValueError("RAFT's adapter workflow supports --method lora or qlora")
    config_file = options.pop("config", options.pop("config_file", None))
    if config_file is None:
        config_file = next((p for p in (paths.root / "opbdh.json", paths.root / ".opbdh.json") if p.is_file()), None)
    base = opbdh.configure(config_file=config_file, **options)
    root = paths.hf_run_path
    project = ft.load_finetune_project(root)
    if project is None:
        project = ft.FineTuneProject(
            model_type="chat", provider=base.provider, gpu_count=base.gpu_count,
            max_dollars_per_hour=base.max_dollars_per_hour, max_spend_dollars=base.max_spend_dollars,
        )
    if recipe:
        if not project.recipes:
            project.active_recipe = ft.validate_recipe_name(str(recipe))
        elif recipe in project.recipes:
            ft.activate_finetune_recipe(project, str(recipe))
        else:
            ft.create_finetune_recipe(project, str(recipe), method=method or "lora")
    if method:
        ft.set_finetune_method(project, method)
    if project.method not in ("lora", "qlora"):
        raise ValueError("Select a LoRA or QLoRA recipe for RAFT's adapter workflow")
    project.model_id = model
    project.model_type = "chat"
    for key, value in training.items():
        setattr(project, key, value)
    for key in ("provider", "gpu_count", "vram_gb", "max_dollars_per_hour", "max_spend_dollars"):
        if key in options or (key == "max_spend_dollars" and "max_spend" in options):
            setattr(project, key, getattr(base, key))
    if "epochs" not in training and os.environ.get("RAFT_EPOCHS"):
        project.epochs = float(os.environ["RAFT_EPOCHS"])
    ft.validate_project(project)
    examples = ft.read_examples([source], source_format="openai")
    if not examples:
        raise ValueError("the RAFT training file contains no examples")
    # This managed copy tracks the current RAFT dataset; never accumulate stale
    # answers from an older generation. Native recipes and job history persist.
    project.source_files = [str(source)]
    ft.replace_project_examples(root, project, examples)
    job = ft.prepare_finetune_job(root, project)
    return root, project, job, base


def prepare_run_dir(dataset: DatasetLike, model: str) -> str:
    """Prepare the native OPBDH training job without renting compute."""
    return str(prepare_finetune(dataset, model, {})[2].directory)


def pick_model_interactively(opbdh: Any) -> str:
    """Pick a huggingface model, optionally searching the hub via opbdh."""
    while True:
        if confirm("Search huggingface for a model?", default=True):
            query = ask("Search query", "instruct")
            try:
                for model_id in opbdh.search_models(query, limit=15):
                    hx.say(f"  {model_id}")
            except Exception as e:
                hx.warn(f"search unavailable: {e}")
        model = ask("Huggingface model id (e.g. Qwen/Qwen2.5-7B-Instruct)")
        if "/" in model:
            return model
        hx.warn("that doesn't look like a huggingface id (expected org/name)")


def run_hf_finetune(
    dataset: DatasetLike,
    model: str,
    opbdh_args: Optional[List[str]] = None,
    interactive: bool = False,
) -> str:
    """
    Launch a huggingface finetune on a GPU pod via opbdh.

    Args:
        dataset: Dataset name or project paths.
        model (str): Huggingface model id (asked interactively if empty).
        opbdh_args (Optional[List[str]]): Native SFT recipe settings and GPU configuration
            overrides (on top of any opbdh.json config).
        interactive (bool): Ask for missing settings instead of relying
            on opbdh defaults/config.

    Returns:
        str: The local path of the trained LoRA adapter.
    """
    opbdh = load_opbdh()
    if not model:
        if not interactive:
            bail("no model given -- pass --model <org/name> or drop --no-interactive")
        model = pick_model_interactively(opbdh)

    from opbdh import finetune as ft
    overrides = parse_opbdh_flags(list(opbdh_args or []))
    dry_run = overrides.pop("dry_run", False)
    if not isinstance(dry_run, bool):
        raise ValueError("--dry-run must be a boolean")
    if interactive:
        if "provider" not in overrides:
            choice = choose("GPU provider", ["use configured provider", "RunPod", "Prime Intellect (multi-cloud marketplace)"])
            if choice:
                overrides["provider"] = ("runpod", "primeintellect")[choice - 1]
        if "method" not in overrides:
            method = choose("Fine-tuning method", ["LoRA", "QLoRA (4-bit)"])
            overrides["method"] = ("lora", "qlora")[method]
        if "max_spend" not in overrides and "max_spend_dollars" not in overrides:
            spend = ask("Max total spend in $ (empty = saved setting)", "")
            if spend:
                overrides["max_spend"] = float(spend)
    root, project, job, base = prepare_finetune(dataset, model, overrides)
    resources = ft.estimate_finetune_resources(project)
    config = ft.build_finetune_run_config(base, root=root, project=project, job=job, resources=resources)
    hx.step(f"{project.method.upper()} via {config.provider}: {job.example_count} examples, "
            f"{config.gpu_count} GPU(s), {resources.vram_per_gpu_gb} GB VRAM/GPU")
    hx.say(f"Native recipe saved in {root / '.opbdh'}; resume with `opbdh ft` from {root}")
    settings = asdict(config)
    if dry_run:
        opbdh.launch(**settings, dry_run=True)
        return ""
    plan = opbdh.plan(**settings)
    summary = opbdh.summarize(plan)
    hx.step(f"launching via {config.provider} on {summary['gpu_candidates'][0]} "
            f"(~${summary['estimated_hourly_dollars']}/hr, "
            f"max spend ${summary['max_spend_dollars']})")
    try:
        result = opbdh.launch(
            **settings, progress=True,
            on_event=lambda event: hx.say(f"[{event.kind}] {event.message}"),
        )
    except opbdh.MaxSpendReached as e:
        bail(f"stopped by the spend guard: {e}")
    except RuntimeError as e:
        bail(f"the finetune failed on the pod: {e}")
    if result is None:
        raise RuntimeError("opbdh returned no training result")
    adapter = Path(result.outputs_dir) / "model"
    if not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(f"opbdh completed without a synced adapter at {adapter}")
    hx.ok(f"done — the LoRA adapter is in {adapter}")
    return str(adapter)
