# RAFT: Retrieval-Augmented Fine-Tuning

Note from [@lumpenspace](http://x.com/lumpenspace):

> This technique is something ive been working last summer/fall, originally planning to get a paper out of it. Then it seemed obvious so i didn't, and instead used
> pieces of this repo for other projects and abandoned this repo.
>
> I [discovered not without horror](https://x.com/lumpenspace/status/1769809977030426831?s=20) that some of the tech is still cutting edge, so i might as well share it.
>
> In this old version, the main simulee was Gary Marcus; the idea was to make a model that could pass as him in a conversation and demonstrate how stochastic
> parrots are still plenty capable to mimic the deterministic ones, but there's a couple interesting tidbits that i've moved to more decent repos but, given my 
> pretty annoying habit of not sharing subpar code, you might as well start here.
>
> Scroll to [usage and functionality](#usage-and-functionality) for cli options, what you can do (apart from what's described below) is automagically fetch, chunk, embed, story, and query a db starting from a substack url.
> 
> Not guaranteeing anything works, but it's a good starting point for a lot of things and includes a couple of new ideas.

### OH GOSH

ok then, a friend asked so now it is more lenient with the version number and uses poetry for the dependencies. It's still a mess, but it's a more runnable mess.

(2.1: poetry is gone — it's uv + hatchling now, like the other repos in this constellation.)

## 2.8

RAFT rethought for thinking models — the reasoning is in
[docs/RAFT-2026.md](docs/RAFT-2026.md); the short version:

- **Recall lives in the thinking phase.** With `raft ft:gen --thinking` (or
  the prep phase's question) each training reply opens a `<think>` block with
  the recalled earlier writing and conversations, continues with the
  reasoning that leads from that recall to the reply — written after the fact
  from the reply the target actually gave — and then gives the reply. Without
  `--thinking`, the recall stays a system note between question and reply, as
  before, for models without a thinking phase and the OpenAI API.
- **One conversation, one date.** A transcript is now one conversation (a
  thread branch, a reply chain, an interview) with its own date and setting.
  The system prompt says who the persona is, what day it is and where the
  conversation happens; retrieval for it is limited to writing dated strictly
  before it. Tweet mode and the LessWrong source write one transcript per
  thread; each exchange is one training example, with the conversation so
  far as prior turns.
- **Conversations are memories too.** Each exchange the pipeline processes is
  remembered, dated, with its question and answer, so later conversations can
  recall earlier ones. Benchmark conversations and live chat never are.
- **2023's limits are knobs.** Examples pack up to 8192 tokens
  (`RAFT_MAX_EXAMPLE_TOKENS`); the fixed rate-limit sleeps became `RAFT_PACE`
  (default 0); the system prompt no longer mentions a function that will be
  "called automatically".
- **Recall is quote-anchored** (2.8.2). A recollection must cite a verbatim
  sentence of the material it comes from, or it is dropped: retrieval always
  returns *something*, and a small summariser will otherwise invent what the
  persona "argued" in a document that says nothing of the kind.
- **Three roles, three endpoints.** The persona model answers through
  `OPENAI_BASE_URL`; the helper LLM that writes summaries and reasoning traces
  can sit elsewhere (`RAFT_LLM_BASE_URL`, `RAFT_LLM_MODEL`, and
  `RAFT_REASONING_MODEL` for a stronger reasoner); embeddings too
  (`RAFT_EMBEDDING_BASE_URL`, `RAFT_EMBEDDING_MODEL`). Unset, each falls back
  to the OpenAI defaults — so a finetune served by `mlx_lm.server` can sit next
  to an ollama doing recall.

## 2.7

**LessWrong as a source.** The gather phase of `raft interactive` can now draw
on any ForumMagnum forum — LessWrong (which also covers the Alignment Forum),
the EA Forum, or another site by URL — through its public GraphQL API, no key
needed. Name the user and choose what the forum feeds: posts (and quick takes)
become grounding documents, comment threads become conversations, or both.
For each comment by the target, whatever it replied to is the questioner's
side — the post's title, author and opening for a top-level comment, the chain
of parent comments otherwise — and the target's comment is the answer. A
branch where the target and an interlocutor go back and forth becomes one
multi-turn exchange, and every comment by the target is an answer exactly
once. Top-level comments on the target's own posts are the target talking to
themselves and go to grounding instead. The importer works newest first and
stops once it has the number of conversations you asked for (default 200),
optionally skipping comments below a karma threshold.

**Training on your own Mac (or CUDA box).** `raft ft:run --model <org/name>
--target mps` (or `--target cuda`) trains the same native opbdh recipe on this
machine's accelerator through opbdh's local execution (opbdh ≥ 1.8: a free-
memory check first, `OPBDH_DEVICE` set for the runner) instead of renting a
pod; the train phase of `raft interactive` offers it as a third venue. The
runner needs the training stack in this environment: `pip install
'raft-ft[local]'`. On Apple Silicon it trains LoRA in fp32 (QLoRA is
CUDA-only), so budget about twice the VRAM estimate. A project directory is
self-contained, so the way to use the big Mac down the hall is to rsync the
project there and run `raft ft:run` on it.

**One endpoint for the prep phase.** The embedding model and the memory
summarizer's model are configurable: `RAFT_EMBEDDING_MODEL` (default
`text-embedding-ada-002`) and `RAFT_LLM_MODEL` (default `gpt-4o`, which the
conversation structurer already used; the summarizer used to hardcode
`gpt-4`). Together with `OPENAI_BASE_URL` this points the whole prep at any
OpenAI-compatible server — ollama with `nomic-embed-text` and a local chat
model, say — so building a dataset needs no OpenAI account. Keep one
embedding model per collection: vectors only compare within it.

## 2.5

Start a persona project with `raft init my-persona`, then `cd my-persona`
and `raft interactive`. Commands inside the project no longer need a dataset
name. Existing `raft <action> <name>` datasets still work.

At the beginning, select all the sources you have: tweets (X / Bluesky),
Substack, blogs / RSS, web pages, PDFs, local files, chat logs, or LessWrong
(2.7). You can
select multiple sources, including several of the same kind. For each, choose
**conversations** or **grounding documents** before importing. Tweets also
support an automatic split: replies become conversations and other posts become
grounding. Conversation sources must contain actual exchanges; an essay is not
turned into invented dialogue. Extracting unstructured conversations uses the
configured LLM.

Grounding is optional. With conversations alone, prep skips chunking and
embedding and creates training examples without retrieved memories.

Projects store `raft.json` alongside `fetch/`, `blobs/`, `metadata/`,
`conversations/`, and `corpus/`. Model state and source choices live in
`metadata/state.json`.

## 2.3

`raft interactive` grew into a five-phase session, resumable per dataset —
what exists on disk (plus `data/{name}_meta.json`) tells it where you left
off, and it suggests the next phase:

1. **gather** — documents and conversations, one source at a time, combined
   into one dataset. New document sources beside substack / tweets / local
   files: any **RSS/Atom feed** (a plain site URL works too — raft follows its
   `rel=alternate` feed link, and teaser-only entries get their linked page
   fetched in full), **single URLs**, and **PDFs**
   (`pip install 'raft-ft[pdf]'`). Re-adding a source only imports what is
   new (deduplicated by link).
2. **prep** — chunk + embed, then generate the finetune examples. Retrieval
   now only surfaces the target's *earlier* writings: chunks carry a
   comparable `date_num`, and each exchange's memory query is filtered to
   documents dated before the interview; unknown-dated documents stay
   retrievable. A collection embedded before 2.3 has no `date_num`, so raft
   warns and skips the filter — re-running `raft embed` (embedding is now an
   upsert) backfills it and turns the filter on.
3. **train** — pick the venue (the OpenAI finetuning API, or a huggingface
   model on a GPU pod via opbdh) and the model. While the job runs, raft
   collects **test questions**, showing for each which documents and tweets
   retrieval will put in the persona's context; the finetuned model id (or
   adapter path) is recorded in the dataset meta.
4. **eval** — generate the benchmark files (when a benchmark transcript
   exists) and run the stored test questions against the finetuned model,
   retrieval context shown alongside each answer.
5. **serve** — also standalone as **`raft serve <name>`**: chat with the
   persona, retrieval-augmented, every turn showing what landed in context
   (answers on stdout, chrome on stderr, so it pipes). OpenAI finetunes are
   served directly; for a LoRA adapter raft prints a serving recipe instead.

## 2.0

New major version. Substack is no longer the only way in:

- **`raft interactive`** — guided end-to-end session. Asks who the target is, collects
  text sources (substack / tweets / local files) and conversation examples. Structured
  inputs (raft transcripts, chat-message JSON, grounding jsonl) are recognised and
  imported as-is; unstructured ones (raw chat logs, podcast transcripts, whatever) are
  converted into transcript datasets with an LLM (`RAFT_LLM_MODEL`, default `gpt-4o`).
  Then chunk/embed/ft:gen/ft:run, each step optional.
- **`raft tweets`** — tweet mode. First asks which network(s) to draw from — **X /
  Twitter**, **Bluesky**, or **both, merged into one dataset** — then calls
  [ariadne](https://github.com/lumpenspace/ariadne)'s Python API to reconstruct reply
  branches and imports them: thread texts become grounding documents, reply branches
  become q/a transcripts. The target's own posts become the answers, whoever they were
  replying to becomes the questioner. For X you choose the source (archive export,
  CSV/JSON dump, or a public handle) and can add the **Community Archive**
  (community-archive.org — no key, and it completes reply threads whose parents were
  authored by other people) and/or a **twitterapi.io** key. Bluesky needs nothing but a
  handle. Needs ariadne (≥ 0.4), published on PyPI as ariadne-x (the bare name is taken by
  the GraphQL library): `pip install ariadne-x`.
- **`raft ft:run <name> --model <model>`** — model routing. OpenAI-finetunable ids
  (gpt-4o-mini and friends) go through the OpenAI finetuning API as before. Any other
  model — i.e. a huggingface `org/name` id — is trained on a rented GPU pod via
  [opbdh](https://github.com/lumpenspace/opbdh)'s native finetuning facility (≥ 1.10.0).
  RAFT imports the generated examples into a resumable recipe; opbdh creates the SFT
  runner, estimates resources per GPU, launches training, and retrieves the adapter.
  Choose **RunPod** or **Prime Intellect's multi-cloud GPU marketplace**: there are
  many GPU and cloud options beyond RunPod, through these two provider backends.

  ```bash
  pip install -U 'raft-ft[hf]'
  opbdh config wizard
  raft ft:run garymarcus --model Qwen/Qwen2.5-7B-Instruct --provider primeintellect --method qlora --max-spend 5
  ```

  Or, since 2.7, on this machine's own accelerator instead of a pod:

  ```bash
  pip install -U 'raft-ft[local]'
  raft ft:run garymarcus --model Qwen/Qwen2.5-7B-Instruct --target mps --epochs 1
  ```

  Native recipe options include `--epochs`, `--learning-rate`, `--max-length`,
  `--batch-size`, `--gradient-accumulation`, and `--recipe`. GPU options include
  `--gpu-count`, `--vram-gb`, and `--max-dollars-per-hour`; a project-root
  `opbdh.json` works too. `--dry-run` previews the launch without renting compute
  or recording a trained model. RAFT supports LoRA and QLoRA adapters.
  The command prints the native recipe directory, where `opbdh ft` can resume the
  workflow, and returns the synced `model/` adapter directory after training.

Both integrations go through the two tools' Python APIs rather than shelling out, so
raft gets the reconstructed threads and the run result as data — and surfaces their
errors (spend guard tripped, remote job failed) directly.

# RAFT / RATF

- [RAFT: Retrieval-Augmented Fine-Tuning](#raft-retrieval-augmented-fine-tuning)
    - [OH GOSH](#oh-gosh)
- [RAFT / RATF](#raft--ratf)
  - [Abstract](#abstract)
  - [Process](#process)
    - [Retrieval-Augmented Fine-Tuning](#retrieval-augmented-fine-tuning)
    - [Generation](#generation)
  - [Usage and Functionality](#usage-and-functionality)
    - [Installation](#installation)
    - [Usage](#usage)
  - [Licence](#licence)

RAFT, or Retrieval-Augmented Fine-Tuning, is a method comprising of a fine-tuning and a RAG-based retrieval phase. It is particularly suited for the creation of agents that realistically emulate a specific human target.

RATF, or Replica Agent Testing Framework, is a framework for evaluating the performance of dialogue agents emulating real-world targets.

## Abstract

The emulation of specific humans in conversational agents presents unique challenges and opportunities for contextual understanding, theory of mind and personalization. In this paper, we introduce the Retrieval-Augmented Fine-Tuning (RAFT) methodology, designed explicitly for simulating individual humans.

RAFT employs a dual-phase process:

In the **Retrieval-Augmented Fine-Tuning phase** proper, combines interview transcripts featuring the human target with appropriately selected, rephrased and evaluated "memories" from the author's past output to give the model a sense of the way the target human combines past writings with the current context to generate responses.

In the **generation phase**, these memories augment the language model's responses to create a nuanced and personalized dialogue.

We demonstrate the efficacy of RAFT through a unique evaluation metric, RATF (Replica Agent Testing Framework) that compares model-generated responses with original human responses in an interview setting. Our findings highlight RAFT's potential to significantly advance the field of personalized, context-sensitive conversational agents.

## Process

### Retrieval-Augmented Fine-Tuning

Two datasets are required for the fine-tuning phase:

- A dataset of **interview transcripts** featuring the target human
- A dataset of the **target's past written output** (tweets, essays, etc.)

The interview transcripts used within a RAG-inspired process retreiving "memories" from the target's written output for each of the interviewer's questions. These memories are then rephrased and evaluated in the context of the target user's answer and, if found useful, they are interpolated between question and answer for the fine-tuning phase.

The steps to reproduce this process are as follows:

1. Create a dataset of interview transcripts featuring the target human. Each interview is a separate `data/{name}_transcript_{i}.json` file holding `{"participants": {"q": ..., "a": ...}, "date": ..., "url": ..., "exchanges": [[question, answer], ...]}`. As of 2.0 you don't have to write these by hand: `raft interactive` takes chat-message JSON, ariadne output or plain unstructured transcripts and produces them for you.
2. Create a dataset of the author's past written output — `data/{name}.jsonl`, one `{"title", "link", "date", "content"}` object per line. `raft fetch` builds this from a substack; `raft tweets` from a tweet archive; `raft interactive` from arbitrary local files.
3. Split the past output dataset in chunks of a size suitable for the chosen embedding model (8192 tokens for Openai's text-embedding-ada-002), and collect metadata and embeddings for each chunk.
4. Store the resulting metadata and embeddings in a vector database (we use ChromaDB).

Then, in order to generate a fine-tuning dataset:

1. For each interview, run the RAG process to retrieve memories from the author's past output for each of the interviewer's questions.
2. Ask the model to rephrase each memory in the context of the interviewer's question. The same model and prompt will be used in the generation phase.
3. Evaluate the resulting memory by the question only first, and discard it if it is not considered useful by the model. We apply this first pass separately because, at inference time, we will not have access to the target human's answer.
4. Save the resulting context including question, memory and as many of the previous [question, memory and answers] tuples as possible, up to the maximum context size the finetune allows, as a new finetune sample.

#### Before/after pics (interview/ ft dataset)

![](https://github.com/lumpenspace/raft/blob/main/Screenshot%202024-03-18%20at%2021.05.39.png?raw=true)

### Generation

The fine-tuned model is then used to generate responses to the interviewer's questions. The model is prompted with the question and the rephrased memories, and the resulting response is evaluated using the RATF framework.


## Usage and Functionality

### Installation

The distribution is named `raft-ft` (`raft` was taken on PyPI); the import and the
CLI are still `raft`. Until the first `raft-ft` release lands on PyPI, install from
git:

```bash
pip install git+https://github.com/lumpenspace/raft
```

Once released:

```bash
pip install raft-ft
```

For development, [uv](https://docs.astral.sh/uv/) manages the environment:

```bash
uv sync --extra dev --extra hf
uv run pytest
uv run ruff check .
```

### Usage

```bash
raft -h
```

```

The following actions are available:

- interactive: Guided end-to-end session: sources, conversations, finetune.
- tweets: Build a dataset from tweets via ariadne interactive.
- fetch: Fetch the blog from Substack and store it in the data directory.
- chunk: Chunk the blog into 4096 token pieces and store them in the data directory.
- embed: Create embeddings for the chunks and store them.
- ft:gen: Generate finetune files for the blog.
- ft:run: Run the finetune job (OpenAI, or huggingface via opbdh).
- bench:setup: Setup the benchmark for the blog.
- ask: Ask a question about the blog content.
- serve: Chat with the finetuned persona, retrieval-augmented.
```

## Licence

MIT
