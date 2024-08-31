"""
This module provides helper functions for working with embeddings.
"""

import json
from typing import Dict, Any, List
from chromadb import PersistentClient
from openai import OpenAI

client = OpenAI()


def get_embedding(text: str) -> List[float]:
    """
    Get the embedding for a given text using OpenAI's API.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding vector.
    """
    embedding_object = client.embeddings.create(
        input=text, model="text-embedding-ada-002"
    )
    embedding_vector = embedding_object.data[0].embedding
    return embedding_vector


def get_and_store_embedding(
    exchange: Dict[str, Any], name: str, metadata: Dict[str, Any]
) -> List[float]:
    """
    Get and store the embedding for a given exchange.

    Args:
        exchange (Dict[str, Any]): The exchange data.
        name (str): The name of the collection.
        metadata (Dict[str, Any]): Metadata for the embedding.

    Returns:
        List[float]: The embedding vector.
    """
    print("Metadata:", metadata)
    qs = exchange.get("question", "") or exchange.get("human", "")

    url = metadata.get("url", "")
    id = "".join(c for c in f"{url}{qs[:20]}" if c.isalnum()).lower()

    chroma_client = PersistentClient(path=f"data/{name}")
    collection = chroma_client.get_or_create_collection(name)

    stored_embedding = collection.get(ids=id).get("embeddings")

    if stored_embedding and len(stored_embedding):
        print("Embedding found in db")
        return list(stored_embedding[0])

    print("getting embeddings")
    embedding = get_embedding(qs)

    meta: Dict[str, str] = (
        {
            **metadata,
            **{"participants": ", ".join(metadata["participants"].values())},
        }
        if "participants" in metadata
        else {"source": "participants"}
    )

    collection.add(ids=id, embeddings=embedding, documents=qs, metadatas=meta)

    return embedding


def store_grounding_embeddings(name: str) -> None:
    """
    Store grounding embeddings for a given name.

    Args:
        name (str): The name of the collection and file to process.
    """
    chroma_client = PersistentClient(path=f"data/{name}")
    collection = chroma_client.get_or_create_collection(name)

    sourcefile = f"data/{name}_chunked.jsonl"

    with open(sourcefile, "r") as f:
        for line in f:
            metadata, document = json.loads(line)
            print(f"Storing {metadata['title']}")

            embeddings = get_embedding(document)
            collection.add(
                ids=f"{metadata['title']}_part_{metadata['part']}",
                embeddings=embeddings,
                documents=document,
                metadatas=metadata,
            )
