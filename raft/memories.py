from typing import List, Dict, Union
from enum import Enum
import time
import re
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor
from chromadb import PersistentClient
import tiktoken

from .prompt_manager import PromptManager
from .embeddings_helpers import get_and_store_embedding

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


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

    def get_similar_and_summarize(self, exchange: List, prev_answer: str) -> ExtractedDataType:
        question, answer = exchange
 
        similar_extracts = self.get_similar_extracts(exchange)
        print(question)
        time.sleep(3)

        print('summarizing memories')
        summaries = self.summarize_helpful_memories(question, similar_extracts, prev_answer)

        useful_memories = ""
        for summary in summaries:
            if len(summary["memory"]):
                useful_memories += f"""from {summary['date']}: \n {summary["memory"]}\n\n"""
        # wait 3 seconds
        time.sleep(3)
        return useful_memories

    def get_similar_extracts(self, exchange:dict) -> ExtractedDataType:
        results = self.collection.query(
            query_embeddings=[get_and_store_embedding(exchange, self.name, self.metadata)],
            n_results=3
        )
        extracted_data: ExtractedDataType = [
            {"date": metadata["date"], "document": document }
            for metadata, document
            in zip(results["metadatas"][0], results["documents"][0])]
        return extracted_data
    
    def summarize_memory(self, memory: ExtractedDataType, question: str, prev_answer: str) -> ExtractedDataType:
        summary = PromptManager().summarize_memory(memory["document"], question, prev_answer, author=self.name)
        if re.sub(r'\W+', '', summary).lower() != "skip":
            return { "date": memory["date"], "memory": summary }
        else:
            return { "date": memory["date"], "memory": ""}

    def summarize_helpful_memories(self, question: str, similar_extracts: ExtractedDataType, prev_answer: str):
        with ThreadPoolExecutor() as executor:
            summaries = list(executor.map(self.summarize_memory, similar_extracts, [question]*len(similar_extracts), [prev_answer]*len(similar_extracts)))
        return summaries
