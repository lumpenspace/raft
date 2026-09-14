import os
import re
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


def _answer(text: str, key: str) -> str:
    match = re.search(rf"^\W*{key}\W*:\s*\**\s*(yes|no)", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).lower() if match else ""


def parse_verdict(text: str) -> Tuple[bool, str]:
    """(passed, reason) from a LEADS / PARAPHRASE / LEANS / WHY checklist reply."""
    leads, paraphrase, leans = _answer(text, "LEADS"), _answer(text, "PARAPHRASE"), _answer(text, "LEANS")
    why = re.search(r"^\W*WHY\W*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    reason = " ".join(why.group(1).split()) if why else " ".join(text.split())[:200]
    if not leads:  # no checklist: fall back to a bare PASS/FAIL if there is one
        return text.strip().upper().startswith("PASS"), reason
    problems = []
    if leads == "no":
        problems.append("it does not lead to the reply")
    if paraphrase == "yes":
        problems.append("it restates the reply")
    if leans == "yes":
        problems.append("it leans on the recollection more than the reply does")
    if problems:
        return False, f"{'; '.join(problems)} ({reason})" if why else "; ".join(problems)
    return True, reason


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
                    "link. Think, do not narrate: start from your own reaction to what they said, never "
                    "from a description of it (no 'the commenter', 'the question tackles', 'my reply'). "
                    "No preamble, do not restate the reply, no quotation marks.\n\n"
                    "The voice, on an unrelated topic: Hm, they're taking the meta-analysis as settled. I "
                    "went through those studies in 2014 and the effect sizes fell apart on replication, so "
                    "I don't buy the premise. The real issue is the burden of proof, and that's what I "
                    "want to push on."
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
        Judge a reasoning trace against the reply it is meant to lead to,
        one criterion at a time (a checklist keeps a mid-size judge honest).

        Returns:
            (passed, reason): passed when the trace reaches the reply's
            conclusion and stance, is not a paraphrase of it, and does not
            lean on the recollection more than the reply itself does.
        """
        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    f"You check one training example for a model of {author}. You are given a question put "
                    "to them, what came to mind from their earlier writing, the private reasoning written "
                    "for them, and the reply they actually gave. Answer exactly these four lines:\n"
                    "LEADS: yes or no -- does the reasoning arrive at the reply's conclusion and stance, "
                    "claiming nothing the reply contradicts?\n"
                    "PARAPHRASE: yes or no -- is the reasoning mostly a restatement of the reply?\n"
                    "LEANS: yes or no -- does the reasoning rely on what came to mind more than the reply "
                    "itself does? (no if nothing came to mind, or the reply visibly builds on it)\n"
                    "WHY: one sentence on the main problem, or on how the reasoning reaches the reply."
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
        return parse_verdict(str(response.choices[0].message.content or ""))

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
