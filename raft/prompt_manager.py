from typing import Dict, List

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class PromptManager:
    """Manages prompts for the RAFT project."""

    def __init__(self):
        """Initialize the PromptManager."""
        self.client = OpenAI()

    def get_interview_system_message(
        self, questioner: str, answerer: str, date: str
    ) -> ChatCompletionSystemMessageParam:
        """
        Get the system message for an interview.

        Args:
            questioner (str): The name of the questioner.
            answerer (str): The name of the answerer.
            date (str): The date of the interview.

        Returns:
            ChatCompletionSystemMessageParam: The system message.
        """
        return ChatCompletionSystemMessageParam(
            role="system",
            content=f"{questioner} is interviewing you, {answerer}.\
                It is the {date}.\n\n\
                To better answer the questions, some memories\
                from your past writing will be retrieved if available, by the \
                retrieve_memories function. It will be called automatically.",
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
        Summarize a memory.

        Args:
            memory (str): The memory to summarize.
            question (str): The current question.
            prev_answer (str): The previous answer.
            author (str): The author's name.
            useful_check (bool, optional): Whether to check for usefulness.\
                Defaults to True.

        Returns:
            str: The summarized memory.
        """
        if useful_check:
            instruction = (
                "Decide whether the quote from his blog, or extract"
                + " from previous interview presented here is helpful"
                + " in answering the question. If it is, rephrase it"
            )
        else:
            instruction = "Read this quote and"

        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=f"""\
                    You are helping {author} prepare for an interview.\n \
                    {instruction} from his perspective, in a way that \
                    would be helpful for answering, in one or two \
                    sentences - type it directly, without intro. \
                    If it is not helpful, simply type 'skip'""",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Previous answer, for context:\n {prev_answer}\n\n \
                    Question: {question}\n\n \
                    Memory: {memory}",
            ),
        ]
        response = self.client.chat.completions.create(model="gpt-4", messages=messages)
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
