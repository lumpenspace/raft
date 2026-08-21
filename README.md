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
  handle. Needs ariadne (≥ 0.4), which is not on PyPI (the name is taken by the GraphQL
  library): `pip install git+https://github.com/lumpenspace/ariadne`.
- **`raft ft:run <name> --model <model>`** — model routing. OpenAI-finetunable ids
  (gpt-4o-mini and friends) go through the OpenAI finetuning API as before. Any other
  model — i.e. a huggingface `org/name` id — is trained on a rented GPU pod via
  [opbdh](https://github.com/lumpenspace/opbdh): raft generates a self-contained LoRA SFT
  run directory (script + requirements + dataset) and hands it to `opbdh launch`.
  Interactively it helps you pick the model (`opbdh models search`) and size the pod;
  non-interactively, opbdh settings pass straight through:

  ```bash
  raft ft:run garymarcus --model Qwen/Qwen2.5-7B-Instruct --vram-gb 48 --max-spend 5
  ```

  Anything opbdh accepts (`--provider`, `--max-dollars-per-hour`, ...) can be appended
  and is forwarded to its Python API, and an `opbdh.json` in the project root works too.
  The trained adapter lands in `runpod_results/<run_id>/results/adapter`. Install with
  `pip install 'raft-ft[hf]'` (python ≥ 3.11) plus a one-time `opbdh config wizard`.

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
```

## Licence

MIT
