"""Exercise native opbdh job preparation offline; never rent a GPU."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opbdh import finetune as ft
from raft.hf_finetune import prepare_finetune, run_hf_finetune


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPBDH_CONFIG", str(tmp_path / "config.json"))
    (tmp_path / "data").mkdir()
    source = tmp_path / "data/d_finetune_openai.jsonl"
    source.write_text(json.dumps({"messages": [
        {"role": "system", "content": "Memory: the blue house"},
        {"role": "user", "content": "Where?", "name": "visitor"},
        {"role": "assistant", "content": "At home.", "name": "target"},
    ]}) + "\n")
    return source


def test_native_recipe_preserves_memories_and_replaces_examples(dataset):
    root, project, job, base = prepare_finetune("d", "test/model", {
        "provider": "primeintellect", "gpu_count": 2, "method": "qlora",
        "recipe": "trial", "epochs": 2, "max_spend": 3,
    })
    assert project.method == "qlora"
    assert project.epochs == 2
    assert project.active_recipe == "trial"
    assert "--num_processes 2" in job.command
    row = json.loads((job.directory / "dataset.jsonl").read_text())
    assert row["prompt"][0]["content"] == "Memory: the blue house"
    assert row["completion"] == [{"role": "assistant", "content": "At home."}]
    assert "name" not in row["prompt"][1]
    config = ft.build_finetune_run_config(base, root=root, project=project, job=job,
        resources=SimpleNamespace(vram_per_gpu_gb=24, disk_gb=80, host_ram_per_gpu_gb=16))
    assert config.provider == "primeintellect"
    assert config.gpu_count == 2
    assert config.vram_gb == 24
    assert config.max_spend_dollars == 3
    dataset.write_text(dataset.read_text().replace("At home.", "Elsewhere."))
    _, saved, job, _ = prepare_finetune("d", "test/model", {})
    assert saved.method == "qlora"
    assert job.example_count == 1
    assert "At home." not in (job.directory / "dataset.jsonl").read_text()


def test_dry_run_and_synced_adapter_path(dataset, tmp_path):
    resources = SimpleNamespace(vram_per_gpu_gb=24, disk_gb=80, host_ram_per_gpu_gb=16)
    with patch.object(ft, "estimate_finetune_resources", return_value=resources), \
         patch("opbdh.launch") as launch, patch("opbdh.plan") as plan, \
         patch("opbdh.summarize", return_value={"gpu_candidates": ["demo"],
             "estimated_hourly_dollars": 1, "max_spend_dollars": 5}):
        assert run_hf_finetune("d", "test/model", ["--dry-run"]) == ""
        assert launch.call_args.kwargs["dry_run"] is True
        plan.assert_not_called()
        output = tmp_path / "outputs"
        adapter = output / "model"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        launch.return_value = SimpleNamespace(outputs_dir=output)
        assert run_hf_finetune("d", "test/model") == str(adapter)
        (adapter / "adapter_config.json").unlink()
        with pytest.raises(RuntimeError, match="without a synced adapter"):
            run_hf_finetune("d", "test/model")


def test_reject_full_model_training_in_adapter_workflow(dataset):
    with pytest.raises(ValueError, match="supports --method"):
        prepare_finetune("d", "test/model", {"method": "full"})


def test_local_target_trains_through_opbdh_launch_local(dataset, tmp_path):
    import os
    import sys

    resources = SimpleNamespace(vram_per_gpu_gb=24, disk_gb=80, host_ram_per_gpu_gb=16)

    def launch_local(argv, *, target, required_gb, cwd, env):
        results = tmp_path / env["OPBDH_RESULTS_DIR"] if not os.path.isabs(env["OPBDH_RESULTS_DIR"]) else __import__("pathlib").Path(env["OPBDH_RESULTS_DIR"])
        (results / "model").mkdir(parents=True)
        (results / "model" / "adapter_config.json").write_text("{}")
        launch_local.calls.append((argv, target, required_gb, cwd))

    launch_local.calls = []
    with patch.object(ft, "estimate_finetune_resources", return_value=resources), \
         patch("opbdh.launch_local", side_effect=launch_local), patch("opbdh.launch") as launch, \
         patch("raft.hf_finetune.require_local_stack"):
        assert run_hf_finetune("d", "test/model", ["--target", "mps", "--dry-run"]) == ""
        assert launch_local.calls == []
        adapter = run_hf_finetune("d", "test/model", ["--target", "mps", "--epochs", "1"])
    launch.assert_not_called()
    (argv, target, required_gb, cwd), = launch_local.calls
    assert target == "mps" and required_gb == 48  # fp32 on MPS: twice the bf16 estimate
    assert argv == [sys.executable, str(cwd / "run.py")] and (cwd / "dataset.jsonl").exists()
    assert adapter.endswith("/model") and json.loads(open(cwd / "config.json").read())["method"] == "lora"


def test_local_target_rejects_unknown_accelerator(dataset):
    with pytest.raises(SystemExit):
        run_hf_finetune("d", "test/model", ["--target", "tpu"])


QWEN38_STYLE = """{%- for message in messages %}
    {%- set content = message.content|trim %}
    {%- if message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}"""


def test_thinking_template_patch_recovers_reasoning_from_the_reply():
    from raft.hf_finetune import thinking_chat_template

    patched = thinking_chat_template(QWEN38_STYLE)
    assert patched and "'</think>' in content" in patched
    jinja2 = pytest.importorskip("jinja2")
    render = jinja2.Environment().from_string(patched).render
    messages = [{"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "<think>\nRecalling: x.\n\nSo y.\n</think>\n\nBecause."}]
    full = render(messages=messages, add_generation_prompt=False)
    prompt = render(messages=messages[:1], add_generation_prompt=True)
    assert full == "<|im_start|>user\nWhy?<|im_end|>\n<|im_start|>assistant\n<think>\nRecalling: x.\n\nSo y.\n</think>\n\nBecause.<|im_end|>\n"
    assert full.startswith(prompt)  # trl masks the prompt as a token prefix
    # a native reasoning_content field still wins, and templates that already split are left alone
    assert "native" in render(messages=[{"role": "assistant", "content": "r", "reasoning_content": "native"}])
    assert thinking_chat_template(patched) is None
    assert thinking_chat_template("{{ messages[0].content }}") is None


def test_thinking_datasets_ship_the_template_with_the_job(dataset, tmp_path):
    from raft import state
    from raft.hf_finetune import install_thinking_template

    state.update_meta("d", thinking=True)
    with patch("raft.hf_finetune.fetch_chat_template", return_value=QWEN38_STYLE):
        _, project, job, _ = prepare_finetune("d", "test/model", {}, local=True)
    config = json.loads((job.directory / "config.json").read_text())
    assert config["chat_template"] == str(job.directory / "chat_template.jinja")
    assert "'</think>' in content" in (job.directory / "chat_template.jinja").read_text()
    with patch("raft.hf_finetune.fetch_chat_template", return_value=QWEN38_STYLE):
        install_thinking_template(job.directory, "test/model", local=False)
    assert json.loads((job.directory / "config.json").read_text())["chat_template"] == "/opbdh-run/user/chat_template.jinja"
    # a model whose template already handles the block needs nothing shipped
    with patch("raft.hf_finetune.fetch_chat_template", return_value="{% if '</think>' in content %}{% endif %}"):
        assert install_thinking_template(job.directory, "test/model", local=True) is None


def test_hybrid_attention_models_get_fused_kernels_on_the_pod(dataset):
    from raft.hf_finetune import add_job_requirements

    with patch("raft.hf_finetune.model_type_of", return_value="qwen3_5"):
        _, project, job, _ = prepare_finetune("d", "Qwen/Qwen3.8-27B", {"extra_requirements": "einops,"})
    lines = (job.directory / "requirements.txt").read_text().split()
    assert "flash-linear-attention" in lines and "einops" in lines and lines.count("flash-linear-attention") == 1
    with patch("raft.hf_finetune.model_type_of", return_value="qwen3_5"):
        assert add_job_requirements(job.directory, "Qwen/Qwen3.8-27B") == []  # already there
    with patch("raft.hf_finetune.model_type_of", return_value="llama"):
        _, _, job, _ = prepare_finetune("d", "meta/llama", {})
    assert "flash-linear-attention" not in (job.directory / "requirements.txt").read_text()
    with patch("raft.hf_finetune.model_type_of", return_value="qwen3_5"):
        _, _, job, _ = prepare_finetune("d", "Qwen/Qwen3.8-27B", {}, local=True)
    assert "flash-linear-attention" not in (job.directory / "requirements.txt").read_text()  # local stack is its own


def test_vram_floor_raises_the_gpu_class(dataset):
    resources = SimpleNamespace(vram_per_gpu_gb=41, disk_gb=80, host_ram_per_gpu_gb=16)
    with patch.object(ft, "estimate_finetune_resources", return_value=resources), \
         patch("opbdh.launch") as launch, patch("raft.hf_finetune.model_type_of", return_value=""):
        run_hf_finetune("d", "test/model", ["--vram-gb", "141", "--dry-run"])
    assert launch.call_args.kwargs["vram_gb"] == 141
    with patch.object(ft, "estimate_finetune_resources", return_value=resources), \
         patch("opbdh.launch") as launch, patch("raft.hf_finetune.model_type_of", return_value=""):
        run_hf_finetune("d", "test/model", ["--vram-gb", "24", "--dry-run"])
    assert launch.call_args.kwargs["vram_gb"] == 41  # never below the estimate
