import os
from typing import Dict, List

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


# The memory summarizer's model: the same knob as the conversation
# structurer, so one OpenAI-compatible endpoint serves the whole prep.
SUMMARY_MODEL = os.environ.get("RAFT_LLM_MODEL", "gpt-4o")


class PromptManager:
    """Manages prompts for the RAFT project."""

    def __init__(self):
        """Initialize the PromptManager (the API client is created lazily)."""
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def get_interview_system_message(
        self, questioner: str, answerer: str, date: str, context: str = "", thinking: bool = False
    ) -> ChatCompletionSystemMessageParam:
        """
        The system message of a conversation: who the persona is, the date
        (which bounds what it can recall), the setting, and how memories
        reach it -- in its own thinking, or as a note before it replies.
        """
        setting = context or "an interview"
        if thinking:
            how = (
                "Before you reply, think: recall what you have written or said before that bears "
                "on this, and work out your reply from it. Then reply as yourself."
            )
        else:
            how = "Where relevant, earlier writing of yours is recalled for you before you reply."
        return ChatCompletionSystemMessageParam(
            role="system",
            content=f"You are {answerer}. It is {date}. This is {setting}; {questioner} is talking to you. {how}",
        )

    def summarize_memory(
        self,
        memory: str,
        question: str,
        prev_answer: str,
        author: str,
        useful_check: bool = True,
    ) -> str:
        """
        Turn a retrieved passage (earlier writing, or an earlier exchange)
        into a first-person recollection the persona can draw on -- or
        "skip" when it does not bear on the question.
        """
        if useful_check:
            instruction = (
                "Decide whether it bears on the question. If it does, restate its relevant point in "
                "one or two sentences, in the first person, as a recollection you could draw on "
                "(\"I've argued that...\") -- type it directly, no preamble. If it does not, type 'skip'."
            )
        else:
            instruction = (
                "Restate its relevant point in one or two sentences, in the first person, as a "
                "recollection you could draw on -- type it directly, no preamble."
            )

        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    f"You are {author}, about to reply in a conversation. Below is something you "
                    f"wrote or said earlier. {instruction}"
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Your previous reply in this conversation, for context:\n{prev_answer or '(none)'}\n\n"
                    f"Earlier material:\n{memory}"
                ),
            ),
        ]
        response = self.client.chat.completions.create(model=SUMMARY_MODEL, messages=messages)
        return str(response.choices[0].message.content).strip()

    def reasoning_trace(
        self, question: str, answer: str, memories: str, prev_answer: str, author: str
    ) -> str:
        """
        For thinking models: the private reasoning that leads from what the
        persona recalled (and the question) to the reply it actually gave.
        Written after the fact from the real reply, so it teaches how the
        target moves from memory to answer rather than inventing positions.
        """
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    f"You are {author}. You are shown a question put to you, what you recalled of your "
                    "earlier writing, and the reply you actually gave. Write the private reasoning that "
                    "took you from the recollection and the question to that reply: first person, present "
                    "tense, three to six sentences, concrete, in your own voice. No preamble, do not "
                    "restate the reply, no quotation marks."
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Your previous reply in this conversation, for context:\n{prev_answer or '(none)'}\n\n"
                    f"Recalled:\n{memories or '(nothing specific came to mind)'}\n\n"
                    f"Your reply:\n{answer}"
                ),
            ),
        ]
        response = self.client.chat.completions.create(model=SUMMARY_MODEL, messages=messages)
        return str(response.choices[0].message.content).strip()

    def contextualise_memories_for_prompt(
        self, memories: List[Dict[str, str]]
    ) -> List[ChatCompletionMessageParam]:
        """
        Contextualize memories for a prompt.

        Args:
            memories (List[Dict[str, str]]): List of memories.

        Returns:
            List[ChatCompletionMessageParam]: Contextualized memories.
        """
        memories_string = "\n".join(
            [f"[memory]{m['memory']}" for m in memories if m["memory"]]
        )
        if memories_string:
            return [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"I wrote something relevant to this question\
                          in the past:\n{memories_string}",
                )
            ]
        else:
            return []
