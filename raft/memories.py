from typing import List, Dict, Union, Any
from enum import Enum
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
    ChatCompletionFunctionMessageParam,
)

from .prompt_manager import PromptManager
from .embeddings_helpers import get_and_store_embedding

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-4-turbo")


class MetaDataKeyEnum(Enum):
    """Enum for metadata keys."""

    DATE = "date"
    PARTICIPANTS = "participants"
    URL = "url"


ExtractedDataType = List[Dict[str, Union[str, datetime, int, float, bool]]]


class MemoryManager:
    """Manages the retrieval and summarization of memories."""

    def __init__(self, name: str, metadata: Dict[MetaDataKeyEnum, Any]):
        """
        Initialize the MemoryManager.

        Args:
            name (str): The name of the collection.
            metadata (Dict[MetaDataKeyEnum, Any]): Metadata for the collection.
        """
        chroma_client = PersistentClient(path=f"data/{name}")
        self.name = name
        self.collection = chroma_client.get_collection(name)
        self.encoder = encoding.encode
        self.metadata = metadata
        self.openai_client = OpenAI()

    def get_similar_and_summarize(self, exchange: List[str], prev_answer: str) -> str:
        """
        Get similar extracts and summarize them.

        Args:
            exchange (List[str]): The current exchange (question and answer).
            prev_answer (str): The previous answer.
            no_useful_check (bool): Whether to skip the usefulness check.

        Returns:
            str: Summarized useful memories.
        """
        question, _ = exchange

        similar: ExtractedDataType = self.get_similar_extracts(exchange)
        print(question)
        time.sleep(3)
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

    def get_similar_extracts(self, exchange: List[str]) -> ExtractedDataType:
        """
        Get similar extracts from the collection.

        Args:
            exchange (List[str]): The current exchange (question and answer).

        Returns:
            ExtractedDataType: List of similar extracts with metadata.
        """
        # Convert MetaDataKeyEnum keys to strings
        string_metadata = {k.value: v for k, v in self.metadata.items()}

        embedding = get_and_store_embedding(
            {"question": exchange[0]}, self.name, string_metadata
        )
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=5,
            include=["metadatas", "documents", "distances"],
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
        prompt_manager = PromptManager()
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

    def ask_question(self, question: str) -> str:
        """
        Ask a question and get an answer based on similar extracts.

        Args:
            question (str): The question to ask.

        Returns:
            str: The generated answer.
        """
        similar: ExtractedDataType = self.get_similar_extracts([question, ""])
        context = "\n\n".join([str(x.get("document", "")) for x in similar])

        memories = self.get_similar_and_summarize([question, ""], "")

        messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=f"Given these previous blog posts and relevant \
                    memories, how would you answer the user's question \
                    in the style of {self.name}?",
            ),
            ChatCompletionSystemMessageParam(role="system", content=context),
            ChatCompletionFunctionMessageParam(
                role="function", name="retrieve_memories", content=memories
            ),
            ChatCompletionUserMessageParam(role="user", content=question),
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo", messages=messages
        )

        return response.choices[0].message.content or ""
