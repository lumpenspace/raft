from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from openai import ChatCompletion, Embedding, GPT3Encoder
import logging
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

    def store_grounding_embeddings(self, blog_posts: List[Dict]):
        # Iterate over the blog posts
        for post in blog_posts:
            # Reserve some tokens for the title and date
            reserved_tokens = self.encoder.encode(f"Title: {post['title']}\nDate: {post['date']}\nPart: 1\n")
            chunk_size = MAX_EMBEDDING_LENGTH - len(reserved_tokens)

            # Split the post content into chunks
            content_tokens = self.encoder.encode(post['content'])
            chunks = [content_tokens[i:i+chunk_size] for i in range(0, len(content_tokens), chunk_size)]

            logging.info(f"Storing {len(chunks)} chunks for post {post['title']}")

            # Add title, date and part to each chunk and store its embedding in the Chroma DB
            for j, chunk in enumerate(chunks):
                document = f"Title: {post['title']}\nDate: {post['date']}\nPart: {j+1}\n{self.encoder.decode(chunk)}"
                self.collection.add(documents=[document])