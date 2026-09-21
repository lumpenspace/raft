# Outstanding

## Correct the misleading phase 5 serving status

Recorded: 2026-09-21. Status: documented; not implemented.

### Observed behavior

After Hugging Face training, the interactive phase map displays
`○ 5 serve ready (<local adapter path>)`. Selecting serve prints a recipe
for merging the LoRA adapter and hosting it behind an OpenAI-compatible
endpoint, then returns without starting a chat session.

This occurred with the ai-panic persona, whose recorded backend is `hf`
and whose artifact is a PEFT LoRA adapter for `Qwen/Qwen2.5-7B-Instruct`.
The adapter exists, but `OPENAI_BASE_URL` was not configured.

### Source and cause

- `src/raft/flows.py`, `phase_details()`: labels serving `ready` whenever
  `status["model"]` is nonempty. It does not check the backend, endpoint,
  or whether the model can answer requests.
- `src/raft/serve.py`, `is_local_adapter()`: treats the recorded `hf`
  backend as authoritative for adapter detection.
- `src/raft/serve.py`, `run_serve()`: when a local adapter is selected and
  `OPENAI_BASE_URL` is unset, prints `HF_SERVE_RECIPE` and returns before
  constructing the chat client. It does not launch an inference server.
- When `OPENAI_BASE_URL` is set, the guard is bypassed; its presence alone
  is not evidence that the endpoint is reachable or serves that model.

Verified against remote main at `8ad9fd2` and the installed raft-ft 3.1.0
serving implementation. The local versioning branch has the same relevant
code.

### Proposed change

Distinguish a recorded training artifact from serving readiness. For an
HF adapter without an endpoint, show something like
`adapter trained; inference endpoint required`. When an endpoint is
configured but untested, say `endpoint configured; not verified` rather
than claiming readiness. Retain the setup recipe when serving cannot
start, and explain that `--model` must identify the model at that endpoint.

Keep status rendering free of implicit network requests. If a readiness
check is added, make it explicit and distinguish connectivity from a
successful inference request.

### Acceptance checks for a future implementation

- A recorded HF adapter without `OPENAI_BASE_URL` is not labeled ready.
- Selecting serve in that state explains the missing endpoint and returns
  without claiming that chat or a server started.
- A configured but unverified endpoint is not labeled healthy.
- Hosted OpenAI model IDs and projects with no recorded model retain
  accurate status wording.
- Cover the status wording and existing adapter guard with focused tests.

This note does not change serving behavior or validate the printed
merge/conversion recipe end to end.

### Confirmed again tonight: this also blocks testing `serve` through `raft interactive`

Recorded: 2026-09-21 (same day, later that night). Status: reproduced;
not implemented. Same root cause as above, called out separately because
it specifically tripped up testing the `feat/error-handling` branch's
`serve.py` fixes (clean Ctrl+C, deduped warnings, clearer failure
messages) through the interactive wizard, after those same fixes had
already been verified working through standalone `raft serve`.

**What happened:** `raft interactive` was launched bare (no
`OPENAI_BASE_URL` etc. set), then "serve" was picked from the phase menu.
`run_serve()`'s early-return guard checks `os.environ.get("OPENAI_BASE_URL")`
once, at call time -- and `raft interactive` is a single long-lived
process, so there is no way to export that env var *after* launching it
and *before* picking "serve" from the menu. It has to already be set in
the shell that started `raft interactive` itself, exactly like standalone
`raft serve`. This is not obvious from the phase menu (which shows
`5 serve ready`) and cost real time to figure out.

**To actually reach the chat loop through `raft interactive`:**

```
RAFT_LLM_MODEL=persona-mlx \
RAFT_EMBEDDING_BASE_URL=https://api.openai.com/v1 \
RAFT_EMBEDDING_API_KEY="$(security find-generic-password -a "$USER" -s OPENAI_API_KEY -w)" \
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=x \
raft interactive
```
then pick "serve" (5) from the menu -- not `raft interactive` on its own
followed by picking serve.

**Note for whoever implements the proposed fix above:** once phase-status
wording distinguishes "adapter trained" from "serving ready," it's
probably worth a one-line hint in the `HF_SERVE_RECIPE` output itself when
`OPENAI_BASE_URL` is unset -- e.g. "export these before starting
`raft interactive`, not from within it" -- since the menu structure makes
it look like configuration should be possible mid-session.

## Document the Hugging Face training-to-serving handoff

Recorded: 2026-09-21. Status: documented; not implemented.

The live [documentation](https://raft.hyperplex.org/docs/) confirms in
[Train](https://raft.hyperplex.org/docs/#train) that Hugging Face training
records an adapter. However, [Talk](https://raft.hyperplex.org/docs/#talk)
assumes a merged, MLX-converted model or an existing inference endpoint,
without walking through how to produce either from the training artifact.

The Talk paragraph also groups `serve` and `comment` together:

- It describes in-process loading through `RAFT_MLX_MODEL`, but that path
  belongs to `raft comment`; `run_serve()` does not read that variable.
  The [endpoint table](https://raft.hyperplex.org/docs/#endpoints) correctly
  scopes the variable to `comment`.
- It describes `/v1/completions`, while `raft serve` calls
  `MemoryManager.ask_question()`, which uses `/v1/chat/completions`.

Proposed documentation improvements:

- Explicitly distinguish the adapter artifact from full base weights and
  a merged model, even when the output directory is named `results/model`.
- Give separate, tested setup instructions for `serve` and `comment`,
  including their endpoint types and model-name selection.
- Show how to keep retrieval embeddings and helper LLM configuration
  separate from the persona inference endpoint.
- Explain the memory requirements of merging before quantization. A small
  adapter does not imply a small merge: the printed PEFT recipe loads the
  base weights before MLX quantization takes place. `device_map="cpu"`
  does not remove that RAM requirement.
- Include a remote serving route for machines that cannot fit the base
  model, and distinguish it from producing a quantized artifact elsewhere
  for later local inference. Compatible servers may load the base plus
  adapter directly without a separate merge.

Concrete case: ai-panic uses Qwen2.5-7B-Instruct; its published weight
[index](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/model.safetensors.index.json)
reports 15,231,233,024 bytes of weights (about 15.2 GB / 14.2 GiB), before
runtime and merge overhead. The user's checked machine has 8 GB unified
memory. The full-precision local merge recipe is not a practical default
for that machine. Quantized inference has different requirements and must
be checked with context/KV-cache and other process memory included.

Acceptance: a reader starting with only the downloaded PEFT adapter can
choose a feasible deployment route and reach a verified response without
mistaking training completion for serving readiness. No remote service
or local conversion has been launched or validated as part of this note.
