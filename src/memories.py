from typing import List, Tuple, Dict, Union
from datetime import datetime
import time
import re
from concurrent.futures import ThreadPoolExecutor
from .embeddings_helpers import get_embedding
from chromadb import PersistentClient
import tiktoken
from prompts import summarize_memory

MAX_EMBEDDING_LENGTH = 2048
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

ExtractedDataType = List[Dict[str, Union[str, datetime]]]


class MemoryManager:
    def __init__(self, name: str):
        chroma_client = PersistentClient(path=f"data/{name}")
        self.name = name
        self.collection = chroma_client.get_collection(name)
        self.encoder = encoding.encode

    def get_similar_and_summarize(self, question: str, prev_answer: str) -> ExtractedDataType:
        similar_extracts = self.get_similar_extracts(question)
        print(question)
        print(similar_extracts)
        summaries = self.summarize_helpful_memories(question, similar_extracts, prev_answer)

        useful_memories = ""
        for summary in summaries:
            if len(summary["memory"]):
                useful_memories += f"""from {summary['date']}: \n {summary["memory"]}\n\n"""
        # wait 3 seconds
        time.sleep(3)
        return useful_memories

    def get_similar_extracts(self, text: str) -> ExtractedDataType:
        results = self.collection.query(
        query_embeddings=[get_embedding(text)],
        n_results=3
        )
        extracted_data: ExtractedDataType = [
            {"date": metadata["date"], "document": document}
            for metadata, document
            in zip(results["metadatas"][0], results["documents"][0])]
        return extracted_data
    
    def summarize_memory(self, memory: ExtractedDataType, question: str, prev_answer: str) -> ExtractedDataType:
        summary = summarize_memory(memory["document"], question, prev_answer, author=self.name)
        if re.sub(r'\W+', '', summary).lower() != "skip":
            return { "date": memory["date"], "memory": summary }
        else:
            return { "date": memory["date"], "memory": ""}

    def summarize_helpful_memories(self, question: str, similar_extracts: ExtractedDataType, prev_answer: str):
        with ThreadPoolExecutor() as executor:
            summaries = list(executor.map(self.summarize_memory, similar_extracts, [question]*len(similar_extracts), [prev_answer]*len(similar_extracts)))
        return summaries
