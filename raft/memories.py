from typing import List, Dict, Union
from enum import Enum
import time
import re
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor
from chromadb import PersistentClient
import tiktoken
from openai import OpenAI

from .prompt_manager import PromptManager
from .embeddings_helpers import get_and_store_embedding

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-4-turbo")


class MetaDataKeyEnum(Enum):
    DATE = "date"
    PARTICIPANTS = "participants"
    URL = "url"


ExtractedDataType = List[Dict[MetaDataKeyEnum, Union[str, datetime]]]


class MemoryManager:
    def __init__(self, name: str, metadata: Dict[MetaDataKeyEnum, str]):
        chroma_client = PersistentClient(path=f"data/{name}")
        self.name = name
        self.collection = chroma_client.get_collection(name)
        self.encoder = encoding.encode
        self.metadata = metadata
        self.openai_client = OpenAI()

    def get_similar_and_summarize(
        self, exchange: List, prev_answer: str, no_useful_check: bool = False
    ) -> ExtractedDataType:
        question, answer = exchange

        similar_extracts: ExtractedDataType = self.get_similar_extracts(exchange)
        print(question)
        time.sleep(3)

        print("summarizing memories")
        summaries = self.summarize_helpful_memories(
            question, similar_extracts, prev_answer, no_useful_check
        )

        useful_memories = ""
        for summary in summaries:
            if len(summary["memory"]):
                useful_memories += (
                    f"""from {summary['date']}: \n {summary["memory"]}\n\n"""
                )
        return useful_memories

    def get_similar_extracts(self, exchange):
        embedding = get_and_store_embedding(exchange, self.name, self.metadata)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=5,
            include=["metadatas", "documents", "distances"],
        )

        extracted_data: ExtractedDataType = [
            {
                "date": metadata.get("date", "Unknown date"),
                "document": document,
                "participants": metadata.get("participants", "Unknown participants"),
                "url": metadata.get("url", ""),
            }
            for metadata, document in zip(
                results["metadatas"][0], results["documents"][0]
            )
        ]

        return extracted_data

    def summarize_memory(
        self,
        memory: ExtractedDataType,
        question: str,
        prev_answer: str,
        no_useful_check: bool = False,
    ) -> ExtractedDataType:
        prompt_manager = PromptManager()  # Create an instance here
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
        similar_extracts: ExtractedDataType,
        prev_answer: str,
        no_useful_check: bool = False,
    ):
        with ThreadPoolExecutor() as executor:
            summaries = list(
                executor.map(
                    self.summarize_memory,
                    similar_extracts,
                    [question] * len(similar_extracts),
                    [prev_answer] * len(similar_extracts),
                    [no_useful_check] * len(similar_extracts),
                )
            )
        return summaries

    def ask_question(self, question: str, no_useful_check: bool = False) -> str:
        similar_extracts: ExtractedDataType = self.get_similar_extracts(
            {"question": question, "answer": ""}
        )
        context = "\n\n".join(
            [str(extract.get("document", "")) for extract in similar_extracts]
        )

        # Use get_similar_and_summarize to process the similar extracts
        useful_memories = self.get_similar_and_summarize(
            [question, ""], "", no_useful_check
        )

        messages = [
            {
                "role": "system",
                "content": f"Given these previous blog posts and relevant memories, how would you answer the user's question in the style of {self.name}?",
            },
            {"role": "system", "content": context},
            {
                "role": "function",
                "name": "retrieve_memories",
                "content": useful_memories,
            },
            {"role": "user", "content": question},
        ]

        response: ChatCompletion = self.openai_client.chat.completions.create(
            model="gpt-4o", messages=messages
        )

        return response.choices[0].message.content
