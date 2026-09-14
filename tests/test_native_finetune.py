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
