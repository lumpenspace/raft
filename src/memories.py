from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from openai import Embedding, GPT3Encoder
import json
import chromadb
from ..prompts import summarize_memory

MAX_EMBEDDING_LENGTH = 2048

class MemoryManager:
    def __init__(self, name: str):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_collection(name)
        self.encoder = GPT3Encoder()

    def get_embedding(self, text:str):
        return Embedding.create(text=text)

    def get_similar_and_summarize(self, text:str):
        similar_extract = self.get_similar_extracts(text)
        summary = self.summarize_helpful_memories(text, similar_extract)
        return summary

    def get_similar_extracts(self, text: str) -> List[str]:
        results = self.collection.query(
          query_embeddings=[self.get_embedding(text)],
          min_score=0.5,
          max_results=3
        )
        similar_texts = [result.document for result in results]
        return similar_texts    

    def summarize_helpful_memories(self, question:str, similar_extracts:List[Tuple[str, float]]):
        with ThreadPoolExecutor() as executor:
            summaries = list(executor.map(summarize_memory, similar_extracts, [question]*len(similar_extracts)))
            summaries = [summary for summary in summaries if summary != "skip"]
        return summaries
    
    def store_grounding_embeddings(self, name: str):
        sourcefile = f'{name}_chunked.jsonl'

        with open(sourcefile, 'r') as f:
            for line in f:
                chunk = json.loads(line)
                document = chunk['document']
                metadata = chunk['metadata']

                embedding = self.get_embedding(document)

                self.collection.add(ids=[f"{metadata['title']}_part_{metadata['part']}"], embeddings=[embedding], documents=[document], metadatas=[metadata])