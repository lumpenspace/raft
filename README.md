# RAFT / RATF

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

1. Create a dataset of interview transcripts featuring the target human. Each interview should be a separate file, with the interviewer's questions and the target human's answers separated by a newline. [TODO: update w current method]
2. Create a dataset of the author's past written output.
3. Split the past output dataset in chunks of a size suitable for the chosen embedding model (8192 tokens for Openai's text-embedding-ada-002), and collect metadata and embeddings for each chunk.
4. Store the resulting metadata and embeddings in a vector database (we use ChromaDB).

Then, in order to generate a fine-tuning dataset:

1. For each interview, run the RAG process to retrieve memories from the author's past output for each of the interviewer's questions.
2. Ask the model to rephrase each memory in the context of the interviewer's question. The same model and prompt will be used in the generation phase.
3. Evaluate the resulting memory by the question only first, and discard it if it is not considered useful by the model. We apply this first pass separately because, at inference time, we will not have access to the target human's answer.
4. Save the resulting context including question, memory and as many of the previous [question, memory and answers] tuples as possible, up to the maximum context size the finetune allows, as a new finetune sample.