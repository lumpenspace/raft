"""
Hugging Face fine-tuning through OPBDH's native SFT facility -- on a
rented GPU pod, or on this machine's own accelerator (Apple Silicon /
CUDA) through opbdh's local execution (`--target mps|cuda`).
"""

from dataclasses import asdict
import json
from pathlib import Path
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

from . import hx, state
from .interactive import ask, bail, choose, confirm
from .project import DatasetLike, dataset_paths

OPBDH_INSTALL_HINT = (
    "opbdh is not installed. Install it with:\n"
    "  pip install -U 'opbdh[ft]>=1.10.0'\n"
    "then run `opbdh config wizard` to set up your provider credentials."
)

LOCAL_TARGETS = ("mps", "cuda")
LOCAL_STACK_HINT = (
    "training on this machine needs the runner's stack in this environment:\n"
    "  pip install 'raft-ft[local]'"
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


# Where opbdh mounts the job directory on a pod (its runner is /opbdh-run/user/run.py).
POD_JOB_DIR = "/opbdh-run/user"

# Model families whose attention runs on reference PyTorch kernels unless
# these are installed ("correct but much slower", says transformers -- an
# order of magnitude on a 27B). Added to the job's requirements on CUDA.
KERNEL_REQUIREMENTS = {
    "qwen3_5": ["flash-linear-attention"],
    "qwen3_5_moe": ["flash-linear-attention"],
    "qwen3_next": ["flash-linear-attention"],
}

# The Qwen3.5+ family renders an assistant turn as <think>reasoning</think>
# content, but takes the reasoning only from a separate `reasoning_content`
# field, which opbdh's example format has no room for. This branch, spliced
# into the model's own template, recovers it from a <think> block written
# in the content -- the way the earlier Qwen3 templates already did.
THINK_SPLIT = (
    "{%- elif '</think>' in content %}\n"
    "            {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
    "            {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
    "        "
)
REASONING_FIELD_BLOCK = re.compile(
    r"(\{%-?\s*if message\.reasoning_content is string\s*-?%\}\s*"
    r"\{%-?\s*set reasoning_content = message\.reasoning_content\s*-?%\}\s*)"
    r"(\{%-?\s*endif\s*-?%\})"
)


def fetch_chat_template(model: str) -> str:
    """The chat template of a huggingface model id or local model directory."""
    local = Path(model).expanduser()
    if local.is_dir():
        jinja = local / "chat_template.jinja"
        if jinja.is_file():
            return jinja.read_text(encoding="utf-8")
        config = json.loads((local / "tokenizer_config.json").read_text(encoding="utf-8"))
        return str(config.get("chat_template") or "")
    from huggingface_hub import hf_hub_download  # an opbdh dependency

    try:
        return Path(hf_hub_download(model, "chat_template.jinja")).read_text(encoding="utf-8")
    except Exception:
        config = json.loads(Path(hf_hub_download(model, "tokenizer_config.json")).read_text(encoding="utf-8"))
        template = config.get("chat_template") or ""
        if isinstance(template, list):  # named templates: {"name": ..., "template": ...}
            named = {t.get("name"): t.get("template") for t in template if isinstance(t, dict)}
            template = named.get("default") or next(iter(named.values()), "")
        return str(template)


def model_type_of(model: str) -> str:
    """The model_type from a huggingface model's config.json ("" if unknown)."""
    try:
        local = Path(model).expanduser()
        if local.is_dir():
            config = json.loads((local / "config.json").read_text(encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download

            config = json.loads(Path(hf_hub_download(model, "config.json")).read_text(encoding="utf-8"))
        return str(config.get("model_type") or "")
    except Exception:  # noqa: BLE001 -- a missing config just means no extras
        return ""


def add_job_requirements(job_dir: Path, model: str, extra: Optional[List[str]] = None) -> List[str]:
    """
    Append the fused-kernel packages a model family needs, plus any
    `--extra-requirements`, to the job's requirements.txt. Returns what
    was added.
    """
    wanted = list(KERNEL_REQUIREMENTS.get(model_type_of(model), [])) + [r for r in (extra or []) if r]
    if not wanted:
        return []
    path = job_dir / "requirements.txt"
    present = set(path.read_text(encoding="utf-8").split()) if path.is_file() else set()
    added = [r for r in wanted if r not in present]
    if added:
        with path.open("a", encoding="utf-8") as f:
            f.write("".join(f"{r}\n" for r in added))
        hx.say(f"job requirements: adding {', '.join(added)}")
    return added


def thinking_chat_template(template: str) -> Optional[str]:
    """
    A copy of the template that reads <think>...</think> out of assistant
    content, or None when the template already does (Qwen3, April 2025)
    or has no notion of reasoning at all (then the tags are just text).
    """
    if "'</think>' in content" in template or '"</think>" in content' in template:
        return None
    patched, count = REASONING_FIELD_BLOCK.subn(lambda m: m.group(1) + THINK_SPLIT + m.group(2), template)
    return patched if count else None


def install_thinking_template(job_dir: Path, model: str, local: bool) -> Optional[Path]:
    """
    Ship a think-aware chat template with the job when the model's own
    template needs one, and point the runner at it. Returns the path written.
    """
    try:
        template = fetch_chat_template(model)
    except Exception as e:
        hx.warn(f"could not fetch the chat template of {model}: {e}")
        return None
    patched = thinking_chat_template(template)
    if patched is None:
        return None
    path = job_dir / "chat_template.jinja"
    path.write_text(patched, encoding="utf-8")
    config_path = job_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = str(path) if local else f"{POD_JOB_DIR}/chat_template.jinja"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hx.say(f"thinking format: shipping a think-aware copy of {model}'s chat template with the job")
    return path


# Training flags belong to OPBDH's SFT recipe, not its pod configuration.
TRAINING_FIELDS = {
    "epochs", "learning_rate", "max_length", "per_device_batch_size",
    "gradient_accumulation_steps", "lora_r", "lora_alpha", "lora_dropout",
    "packing", "seed", "trust_remote_code", "chat_template",
}
FLAG_ALIASES = {"batch_size": "per_device_batch_size", "gradient_accumulation": "gradient_accumulation_steps"}


def prepare_finetune(dataset: DatasetLike, model: str, options: Dict[str, Any], local: bool = False):
    """
    Import RAFT examples into a resumable native OPBDH recipe and job.

    For a dataset generated with --thinking, the job also carries a
    think-aware copy of the model's chat template when its own would not
    read the <think> block out of the reply (see install_thinking_template).
    """
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
    extra_requirements = str(options.pop("extra_requirements", "") or "").split(",")
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
    if state.load_meta(paths).get("thinking") and not project.chat_template:
        install_thinking_template(job.directory, model, local)
    if not local:  # the pod installs requirements.txt; this machine's stack is its own business
        add_job_requirements(job.directory, model, extra_requirements)
    return root, project, job, base


def prepare_run_dir(dataset: DatasetLike, model: str) -> str:
    """Prepare the native OPBDH training job without renting compute."""
    return str(prepare_finetune(dataset, model, {})[2].directory)


def default_local_target(opbdh: Any) -> str:
    """The accelerator opbdh finds on this machine: cuda over mps."""
    capacities = opbdh.local_accelerators()
    for target in ("cuda", "mps"):
        if target in capacities:
            return target
    bail("no local accelerator found (opbdh needs torch with CUDA or MPS here)")
    return ""


def require_local_stack() -> None:
    """The runner imports these itself; fail before opbdh's capacity check does."""
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as e:
        bail(f"{LOCAL_STACK_HINT}\n  ({e})")


def run_local_finetune(
    dataset: DatasetLike, model: str, target: str, overrides: Dict[str, Any], dry_run: bool = False
) -> str:
    """
    Train on this machine's accelerator: the same native opbdh recipe and
    runner as a pod run, launched through opbdh.launch_local (which
    checks free accelerator memory first and sets OPBDH_DEVICE).

    Returns:
        str: The local path of the trained LoRA adapter.
    """
    opbdh = load_opbdh()
    from opbdh import finetune as ft

    if target not in LOCAL_TARGETS:
        bail(f"--target must be one of {', '.join(LOCAL_TARGETS)}, not {target!r}")
    if target == "mps":
        # QLoRA's 4-bit kernels are CUDA-only; the runner refuses it there.
        overrides.setdefault("method", "lora")
    root, project, job, _ = prepare_finetune(dataset, model, overrides, local=True)
    resources = ft.estimate_finetune_resources(project)
    # The estimate assumes bf16 on a GPU; on MPS the runner trains in fp32.
    required_gb = resources.vram_per_gpu_gb * (2 if target == "mps" else 1)
    results_dir = root.expanduser().resolve() / getattr(ft, "RESULTS_DIR", "results") / project.active_recipe
    argv = [sys.executable, str(job.directory / "run.py")]
    hx.step(
        f"{project.method.upper()} on this machine ({target}): {job.example_count} examples, "
        f"~{required_gb} GB of accelerator memory"
    )
    hx.say(f"Native recipe saved in {root / '.opbdh'}; results land in {results_dir}")
    if dry_run:
        hx.say("dry run: " + " ".join(argv))
        return ""
    require_local_stack()
    try:
        opbdh.launch_local(
            argv, target=target, required_gb=required_gb, cwd=job.directory,
            env={"OPBDH_RESULTS_DIR": str(results_dir)},
        )
    except ValueError as e:
        bail(f"cannot train on this machine: {e}")
    except subprocess.CalledProcessError as e:
        bail(f"the local finetune failed (exit status {e.returncode})")
    adapter = results_dir / "model"
    if not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(f"the runner finished without an adapter at {adapter}")
    hx.ok(f"done — the LoRA adapter is in {adapter}")
    return str(adapter)


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
            overrides (on top of any opbdh.json config); `--target mps|cuda`
            trains on this machine instead of a pod.
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
    target = overrides.pop("target", None)
    if target is True:
        bail(f"--target needs a value: {' or '.join(LOCAL_TARGETS)}")
    if interactive:
        if target is None and "provider" not in overrides:
            choice = choose(
                "Where should it train?",
                [
                    "use configured provider", "RunPod", "Prime Intellect (multi-cloud marketplace)",
                    "this machine (Apple Silicon / CUDA, via opbdh local execution)",
                ],
            )
            if choice == 3:
                target = default_local_target(opbdh)
            elif choice:
                overrides["provider"] = ("runpod", "primeintellect")[choice - 1]
        if "method" not in overrides and target != "mps":
            method = choose("Fine-tuning method", ["LoRA", "QLoRA (4-bit)"])
            overrides["method"] = ("lora", "qlora")[method]
        if not target and "max_spend" not in overrides and "max_spend_dollars" not in overrides:
            spend = ask("Max total spend in $ (empty = saved setting)", "")
            if spend:
                overrides["max_spend"] = float(spend)
    if target:
        return run_local_finetune(dataset, model, str(target), overrides, dry_run)
    root, project, job, base = prepare_finetune(dataset, model, overrides)
    resources = ft.estimate_finetune_resources(project)
    config = ft.build_finetune_run_config(base, root=root, project=project, job=job, resources=resources)
    # --vram-gb is a floor on the GPU class, not just a capacity: opbdh
    # picks the cheapest card that fits the estimate, and a bigger floor is
    # how you ask for a faster one.
    floor = int(overrides.get("vram_gb") or 0)
    if floor > config.vram_gb:
        config.vram_gb = floor
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
