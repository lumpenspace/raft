import os
from typing import Dict, List, Tuple

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


# The memory summarizer's model: the same knob as the conversation
# structurer, so one OpenAI-compatible endpoint serves the whole prep.
SUMMARY_MODEL = os.environ.get("RAFT_LLM_MODEL", "gpt-4o")
# The reasoning traces are one call per exchange (summaries are five) and
# carry the persona's voice, so they may use a stronger model.
REASONING_MODEL = os.environ.get("RAFT_REASONING_MODEL", SUMMARY_MODEL)

# The previous reply is context, not material: a few hundred characters
# orient the summariser; the full reply would dominate every prompt.
CONTEXT_CHARS = 800


def _context(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= CONTEXT_CHARS else text[:CONTEXT_CHARS].rstrip() + " [...]"


def helper_client() -> OpenAI:
    """An OpenAI-compatible client for the helper LLM (see PromptManager.client)."""
    return OpenAI(
        base_url=os.environ.get("RAFT_LLM_BASE_URL") or None,
        api_key=os.environ.get("RAFT_LLM_API_KEY") or None,
    )


class PromptManager:
    """Manages prompts for the RAFT project."""

    def __init__(self):
        """Initialize the PromptManager (the API client is created lazily)."""
        self._client = None

    @property
    def client(self) -> OpenAI:
        """
        The helper-LLM client (summaries, reasoning traces). RAFT_LLM_BASE_URL
        / RAFT_LLM_API_KEY point it at a different server than the persona
        model is served from; unset, the OpenAI defaults apply.
        """
        if self._client is None:
            self._client = helper_client()
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
        date: str = "",
        title: str = "",
    ) -> str:
        """
        Turn a retrieved passage (earlier writing, or an earlier exchange)
        into a first-person recollection the persona can draw on -- or
        "skip" when it does not bear on the question.

        The reply is quote-anchored: it must cite one verbatim sentence of
        the material (SOURCE) before the recollection (RECALL), so the
        caller can drop recollections the material does not support --
        weak matches otherwise tempt a model to invent what it "argued".
        """
        if useful_check:
            decision = (
                "Decide whether it bears on the question. If nothing in it does, reply with the single "
                "word skip. Otherwise reply in exactly this form:"
            )
        else:
            decision = "Reply in exactly this form:"
        origin = ", ".join(part for part in (date, f'"{title}"' if title else "") if part)

        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    f"You are {author}, about to reply in a conversation. Below is something you wrote or "
                    f"said earlier. {decision}\n"
                    "SOURCE: <one sentence copied verbatim from the earlier material>\n"
                    "RECALL: <one or two sentences, first person, restating that point as a recollection you "
                    "could draw on -- \"I've argued that...\">\n"
                    "Only restate what the material actually says; never infer what you might have thought."
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Your previous reply in this conversation, for context:\n{_context(prev_answer) or '(none)'}\n\n"
                    f"Earlier material{f' ({origin})' if origin else ''}:\n{memory}"
                ),
            ),
        ]
        response = self.client.chat.completions.create(model=SUMMARY_MODEL, messages=messages)
        return str(response.choices[0].message.content).strip()

    def reasoning_trace(
        self, question: str, answer: str, memories: str, prev_answer: str, author: str, objection: str = ""
    ) -> str:
        """
        For thinking models: the private reasoning that leads from the
        question (and whatever came to mind) to the reply actually given.
        Written after the fact from the real reply, so it carries the
        target's position rather than the writer's; the recollection is
        drawn on only as far as the reply itself does, never forced in.
        """
        retry = (
            f"\n\nA previous attempt was rejected: {objection} Write it again so that it leads to the reply."
            if objection else ""
        )
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    f"You are {author}. You are shown a question put to you, what came to mind from your "
                    "earlier writing, and the reply you actually gave. Write the private reasoning that "
                    "took you from the question to that reply, as it went through your head in the "
                    "moment: first person, present tense, three to six sentences, concrete, in your own "
                    "voice. What came to mind may or may not have shaped the reply: draw on it exactly as "
                    "far as the reply does -- if the reply builds on it, show how; if the reply does not, "
                    "leave it aside or note in passing that it is not the point here. Never manufacture a "
                    "link. Think, do not narrate -- never describe the exchange from outside (no 'the "
                    "commenter', 'the original claim', 'my reply'). No preamble, do not restate the reply, "
                    "no quotation marks."
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Your previous reply in this conversation, for context:\n{_context(prev_answer) or '(none)'}\n\n"
                    f"What came to mind:\n{memories or '(nothing specific)'}\n\n"
                    f"Your reply:\n{answer}{retry}"
                ),
            ),
        ]
        response = self.client.chat.completions.create(model=REASONING_MODEL, messages=messages)
        return str(response.choices[0].message.content).strip()

    def check_trace(self, question: str, memories: str, reasoning: str, answer: str, author: str) -> Tuple[bool, str]:
        """
        Judge a reasoning trace against the reply it is meant to lead to.

        Returns:
            (passed, reason): passed when the trace reaches the reply's
            conclusion and stance, contradicts nothing in it, is not a
            paraphrase of it, and does not pretend the reply relies on the
            recollection when the reply shows no sign of that.
        """
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    f"You check one training example for a model of {author}. You are given a question put "
                    "to them, what came to mind from their earlier writing, the private reasoning written "
                    "for them, and the reply they actually gave. The reasoning passes only if all hold: it "
                    "arrives at the reply's conclusion and stance; it claims nothing the reply contradicts; "
                    "it is not merely a paraphrase or restatement of the reply; and it does not lean on the "
                    "recollection more than the reply itself does -- if the reply shows no sign of drawing "
                    "on it, the reasoning must not pretend it did. Answer on the first line with PASS or "
                    "FAIL, then one sentence saying what is off (FAIL) or how the reasoning reaches the "
                    "reply (PASS)."
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"Question: {question}\n\nWhat came to mind:\n{memories or '(nothing specific)'}\n\n"
                    f"Reasoning:\n{reasoning}\n\nReply actually given:\n{answer}"
                ),
            ),
        ]
        response = self.client.chat.completions.create(model=REASONING_MODEL, messages=messages)
        text = str(response.choices[0].message.content or "").strip()
        first, _, rest = text.partition("\n")
        passed = first.strip().upper().startswith("PASS")
        reason = " ".join((first.split(":", 1)[1] if ":" in first else rest).split()).strip() or text[:200]
        return passed, reason

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
