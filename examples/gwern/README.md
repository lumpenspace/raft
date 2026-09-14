# gwern, from LessWrong

A persona built from one source: gwern's LessWrong account. The 200 newest
comment threads become dated conversations (each thread branch is one
conversation, the target's comments are the answers); the posts, the quick
takes and every older comment become grounding — the past writing the persona
recalls while it thinks. No gwern.net scrape: about 11,700 comments over
fifteen years give recall something to cite for almost any topic.

## Bootstrap

```bash
pip install -U raft-ft
examples/gwern/bootstrap.sh                 # creates ./gwern
examples/gwern/bootstrap.sh ~/personas/gwern
```

The script is `raft init` plus one command:

```bash
raft lesswrong --user gwern --conversations 200
```

No API key is needed for this step; it takes a few minutes, mostly polite
delays between pages of the public GraphQL API. The script refuses to run into
a project that already has conversations: the importer appends, so a second
run would duplicate them. `CONVERSATIONS=0` takes every thread; `--min-karma`
and `--role corpus|conversation` are there too (`raft lesswrong -h`).

What you get (September 2026): 11.8k grounding documents, 200 conversations,
247 exchanges, dated 2021–2026. The project's `metadata/state.json` records
the target (the forum display name) so `raft interactive` picks up at prep.

## Prep

The helper models — embeddings, recall summaries, reasoning traces — talk to
any OpenAI-compatible endpoint. `OPENAI_API_KEY` alone works; so does a local
ollama:

```bash
export RAFT_EMBEDDING_BASE_URL=http://localhost:11434/v1 RAFT_EMBEDDING_MODEL=nomic-embed-text
export RAFT_LLM_BASE_URL=http://localhost:11434/v1 RAFT_LLM_MODEL=qwen2.5:14b
cd gwern
raft chunk && raft embed        # ~12k chunks; about half an hour with nomic on a Mac
raft ft:gen --thinking          # recall + reasoning in <think>; 40–60 s per exchange with a 14B
```

A 14B summariser writes usable recall but not reasoning traces that survive
the judge (each trace is checked against the reply it must lead to). Rewrite
those with a stronger model on another endpoint:

```bash
RAFT_LLM_BASE_URL=... RAFT_REASONING_MODEL=... raft ft:gen --rewrite-traces
```

## Train

```bash
pip install -U 'raft-ft[hf]'
opbdh config wizard
raft ft:run --model Qwen/Qwen3.8-27B --provider runpod --method qlora \
  --gpu-count 2 --vram-gb 94 --gradient-accumulation 4 --epochs 2 \
  --max-length 8192 --max-spend 6 --max-dollars-per-hour 8.2
```

Two H100s, about twelve minutes, under two dollars; the QLoRA adapter path is
recorded in `metadata/state.json`. Qwen3.5/3.8 are hybrid-attention models and
need the fused kernels raft ships for them on pods; on Apple Silicon
(`--target mps`) the same run is measured in hours.

## Talk to it

```bash
raft comment --source https://www.lesswrong.com/posts/...   # its comment on a post
raft comment --web 8090                                     # a page that does the same
raft serve                                                  # chat
```

`raft comment` runs the persona through `RAFT_MLX_MODEL` (an MLX conversion of
the merged adapter) or an OpenAI-compatible `/v1/completions` endpoint
(`OPENAI_BASE_URL` plus `--model`), with the recall prefilled into the think
block exactly as in training.
