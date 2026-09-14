from typing import List, Dict, Optional, Union, Any
from enum import Enum
import os
import time
import re
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor
from chromadb import PersistentClient
import tiktoken
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from . import hx
from .prompt_manager import PromptManager
from .embeddings_helpers import get_embedding, store_exchange_embedding
from .sources import date_num
from .project import DatasetLike, dataset_paths

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-4-turbo")

# Seconds to pause between LLM-heavy steps. The 2023 code slept a fixed
# 3 s per exchange for OpenAI's rate limits of the day; against a local
# endpoint (or today's limits) there is nothing to wait for.
PACE = float(os.environ.get("RAFT_PACE", "0"))


class MetaDataKeyEnum(Enum):
    """Enum for metadata keys."""

    DATE = "date"
    PARTICIPANTS = "participants"
    URL = "url"


ExtractedDataType = List[Dict[str, Union[str, datetime, int, float, bool]]]


class MemoryManager:
    """Manages the retrieval and summarization of memories."""

    def __init__(
        self, dataset: DatasetLike, metadata: Dict[MetaDataKeyEnum, Any], client: Optional[OpenAI] = None
    ):
        """
        Initialize the MemoryManager.

        Args:
            dataset: Dataset name or project paths.
            metadata (Dict[MetaDataKeyEnum, Any]): The conversation's
                participants / date / url; the date bounds retrieval.
            client: An OpenAI-compatible client for answering (defaults to
                the environment's).
        """
        self.dataset = dataset_paths(dataset)
        self.name = self.dataset.name
        # The existence check keeps a missing dataset missing: merely
        # constructing a PersistentClient would create the chroma dir.
        self.collection = None
        if (self.dataset.chroma_path / "chroma.sqlite3").exists():
            try:
                chroma_client = PersistentClient(path=str(self.dataset.chroma_path))
                self.collection = chroma_client.get_collection(self.dataset.collection)
            except Exception:
                pass
        if self.collection is None:
            # Degrade to no memories rather than not running at all, but
            # never silently: ungrounded output looks just like grounded.
            hx.warn(
                f"no grounding collection for {self.name} -- proceeding without "
                f"memories (chunk + embed the corpus to enable retrieval)"
            )
        self._dated: Union[bool, None] = None
        self.encoder = encoding.encode
        self.metadata = metadata
        self.openai_client = client or OpenAI()
        self.prompt_manager = PromptManager()

    def get_similar_and_summarize(
        self, exchange: List[str], prev_answer: str, store: bool = True
    ) -> str:
        """
        Get similar extracts and summarize them.

        Args:
            exchange (List[str]): The current exchange (question and answer).
            prev_answer (str): The previous answer.
            store (bool): Also store the question's embedding (see
                get_similar_extracts).

        Returns:
            str: Summarized useful memories.
        """
        question, _ = exchange

        similar: ExtractedDataType = self.get_similar_extracts(exchange, store=store)
        if not similar:
            return ""
        hx.step(" ".join(question.split())[:100])
        if PACE:
            time.sleep(PACE)
        summaries: List[Dict[str, str]] = self.summarize_helpful_memories(
            question, similar, prev_answer
        )

        useful_memories = ""
        for summary in summaries:
            if len(summary["memory"]):
                useful_memories += (
                    f"""from {summary['date']}: \n {summary["memory"]}\n\n"""
                )
        return useful_memories

    def _collection_is_dated(self) -> bool:
        """
        Whether the collection carries date_num metadata at all.

        Collections embedded before 2.3 have none: warn once and skip
        the earlier-writings filter for them (re-running `raft embed`
        upserts date_num everywhere and activates it).
        """
        if self._dated is None:
            got = self.collection.get(where={"date_num": {"$gte": 0}}, limit=1)
            self._dated = bool(got.get("ids"))
            if not self._dated:
                hx.warn(
                    "the embedding store predates date filtering; retrieval "
                    "may surface later writings -- re-run `raft embed` to fix"
                )
        return self._dated

    def get_similar_extracts(
        self, exchange: List[str], store: bool = True
    ) -> ExtractedDataType:
        """
        Get similar extracts from the collection.

        When the interview's date is known, only *earlier* writings are
        retrieved (date_num < the interview's day) -- the target cannot
        remember what they had not yet written. Documents with unknown
        dates carry date_num 0 and stay retrievable. An empty filtered
        result means nothing earlier exists, and stays empty.

        Args:
            exchange (List[str]): The current exchange (question and answer).
            store (bool): Afterwards, remember the exchange (question and
                answer, dated) as a conversation memory for later
                conversations. Dataset generation does; serving and
                benchmarks must not, or their questions would contaminate
                the corpus.

        Returns:
            ExtractedDataType: List of similar extracts with metadata.
        """
        if self.collection is None:
            return []

        # Convert MetaDataKeyEnum keys to strings
        string_metadata = {k.value: v for k, v in self.metadata.items()}

        embedding = get_embedding(exchange[0])
        before = date_num(string_metadata.get("date"))
        query_args = {
            "query_embeddings": [embedding],
            "n_results": 5,
            "include": ["metadatas", "documents", "distances"],
        }
        if before and self._collection_is_dated():
            query_args["where"] = {"date_num": {"$lt": before}}
        results = self.collection.query(**query_args)
        if store:
            # After the query, so an exchange never retrieves itself.
            store_exchange_embedding(
                {"question": exchange[0], "answer": exchange[1] if len(exchange) > 1 else ""},
                self.dataset, string_metadata, embedding,
            )

        extracted_data: ExtractedDataType = [
            {
                "date": metadata.get("date", "Unknown date"),
                "document": document,
                "participants": metadata.get("participants", "Unknown"),
                "url": metadata.get("url", ""),
            }
            for metadata, document in zip(
                results["metadatas"][0] if results["metadatas"] else [],
                results["documents"][0] if results["documents"] else [],
            )
        ]

        return extracted_data

    def summarize_memory(
        self,
        memory: Dict[str, str],
        question: str,
        prev_answer: str,
        no_useful_check: bool = False,
    ) -> Dict[str, str]:
        """
        Summarize a single memory.

        Args:
            memory (Dict[str, str]): The memory to summarize.
            question (str): The current question.
            prev_answer (str): The previous answer.
            no_useful_check (bool): Whether to skip the usefulness check.

        Returns:
            Dict[str, str]: Summarized memory with date.
        """
        prompt_manager = self.prompt_manager
        summary = prompt_manager.summarize_memory(
            memory["document"],
            question,
            prev_answer,
            author=self.name,
            useful_check=not no_useful_check,
        )
        if re.sub(r"\W+", "", summary).lower() != "skip":
            return {"date": memory["date"], "memory": summary}
        else:
            return {"date": memory["date"], "memory": ""}

    def summarize_helpful_memories(
        self,
        question: str,
        similar: ExtractedDataType,
        prev_answer: str,
    ) -> List[Dict[str, str]]:
        """
        Summarize helpful memories from similar extracts.

        Args:
            question (str): The current question.
            similar (ExtractedDataType): List of similar extracts.
            prev_answer (str): The previous answer.
            no_useful_check (bool): Whether to skip the usefulness check.

        Returns:
            List[Dict[str, str]]: List of summarized memories.
        """
        with ThreadPoolExecutor() as executor:
            summaries = list(
                executor.map(
                    lambda x: self.summarize_memory(x, question, prev_answer),
                    similar,
                )
            )
        return summaries

    def reasoning_trace(self, question: str, answer: str, memories: str, prev_answer: str) -> str:
        """For thinking models: the reasoning from recall to the real reply."""
        return self.prompt_manager.reasoning_trace(question, answer, memories, prev_answer, author=self.name)

    def ask_question(self, question: str, model: str = "gpt-4-turbo", thinking: bool = False) -> str:
        """
        Ask a question and get an answer based on similar extracts.

        Args:
            question (str): The question to ask.
            model (str): The model that answers -- typically the
                finetuned persona model.
            thinking (bool): The persona was trained to recall and reason
                in a <think> block; its thinking is shown on stderr and
                only the reply is returned.

        Returns:
            str: The generated answer.
        """
        # store=False: asking must never write the question into the
        # grounding collection it retrieves from.
        memories = self.get_similar_and_summarize([question, ""], "", store=False)
        today = datetime.now().date().isoformat()
        target = self.metadata.get(MetaDataKeyEnum.PARTICIPANTS, {}).get("a") if isinstance(
            self.metadata.get(MetaDataKeyEnum.PARTICIPANTS), dict) else None
        messages: List[ChatCompletionMessageParam] = [
            self.prompt_manager.get_interview_system_message(
                "someone", target or self.name, today, "a conversation", thinking
            ),
        ]
        if memories:
            messages.append(ChatCompletionSystemMessageParam(
                role="system", content=f"Earlier writing of yours that may bear on this:\n{memories}"
            ))
        messages.append(ChatCompletionUserMessageParam(role="user", content=question))

        response = self.openai_client.chat.completions.create(
            model=model, messages=messages
        )
        content = response.choices[0].message.content or ""
        thought = getattr(response.choices[0].message, "reasoning_content", None) or ""
        if "</think>" in content:
            thought, content = content.split("</think>", 1)
            thought = thought.replace("<think>", "")
        if thought.strip():
            hx.say(f"thinking: {' '.join(thought.split())[:1500]}")
        return content.strip()


def preview_context(
    dataset: DatasetLike, question: str, n_results: int = 5
) -> List[Dict[str, str]]:
    """
    What retrieval would put in the persona's context for a question --
    without storing the question or calling the summarizer.

    Returns:
        List[Dict[str, str]]: {"title", "date", "url", "snippet"} rows,
        best match first; empty if nothing is embedded yet.
    """
    # Existence check first: constructing a PersistentClient would
    # create data/{name}/ and make an unembedded dataset look embedded.
    paths = dataset_paths(dataset)
    if not (paths.chroma_path / "chroma.sqlite3").exists():
        return []
    try:
        collection = PersistentClient(path=str(paths.chroma_path)).get_collection(
            paths.collection
        )
    except Exception:
        return []

    results = collection.query(
        query_embeddings=[get_embedding(question)],
        n_results=n_results,
        include=["metadatas", "documents"],
    )
    rows = []
    for metadata, document in zip(
        (results["metadatas"] or [[]])[0], (results["documents"] or [[]])[0]
    ):
        rows.append(
            {
                "title": str(
                    metadata.get("title")
                    or metadata.get("participants")
                    or "past exchange"
                ),
                "date": str(metadata.get("date", "")),
                "url": str(metadata.get("url") or metadata.get("link") or ""),
                "snippet": " ".join(str(document).split())[:140],
            }
        )
    return rows
