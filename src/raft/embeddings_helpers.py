"""
This module provides helper functions for working with embeddings.
"""

import json
import os
from typing import Dict, Any, List
from chromadb import PersistentClient
from openai import OpenAI

from .sources import date_num
from .project import DatasetLike, dataset_paths

# Any model behind an OpenAI-compatible endpoint works (OPENAI_BASE_URL
# pointed at ollama, say, with RAFT_EMBEDDING_MODEL=nomic-embed-text);
# one collection must keep one model, since vectors are only comparable
# within it.
EMBEDDING_MODEL = os.environ.get("RAFT_EMBEDDING_MODEL", "text-embedding-ada-002")

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def get_embedding(text: str) -> List[float]:
    """
    Get the embedding for a given text using OpenAI's API.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding vector.
    """
    embedding_object = _get_client().embeddings.create(
        input=text, model=EMBEDDING_MODEL
    )
    embedding_vector = embedding_object.data[0].embedding
    return embedding_vector


def store_exchange_embedding(
    exchange: Dict[str, Any], dataset: DatasetLike, metadata: Dict[str, Any], embedding: List[float]
) -> None:
    """
    Remember one exchange as a conversation memory: the question and the
    persona's answer, dated, so later conversations can recall it (the
    earlier-writings filter applies to it like to any document).
    """
    question = exchange.get("question", "")
    answer = exchange.get("answer", "")
    participants = metadata.get("participants") or {}
    q_name = participants.get("q", "Q") if isinstance(participants, dict) else "Q"
    a_name = participants.get("a", "A") if isinstance(participants, dict) else "A"
    document = f"{q_name}: {question}\n{a_name}: {answer}" if answer else question

    url = metadata.get("url", "")
    id = "".join(c for c in f"{url}{question[:20]}" if c.isalnum()).lower()

    paths = dataset_paths(dataset)
    paths.chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = PersistentClient(path=str(paths.chroma_path))
    collection = chroma_client.get_or_create_collection(paths.collection)

    meta: Dict[str, Any] = {
        k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))
    }
    if isinstance(participants, dict):
        meta["participants"] = ", ".join(str(v) for v in participants.values())
    meta["kind"] = "exchange"
    # Comparable date for the earlier-writings-only retrieval filter.
    meta["date_num"] = date_num(meta.get("date"))
    collection.upsert(ids=id, embeddings=embedding, documents=document, metadatas=meta)


def store_grounding_embeddings(dataset: DatasetLike) -> None:
    """
    Store grounding embeddings for a given name.

    Args:
        name (str): The name of the collection and file to process.
    """
    paths = dataset_paths(dataset)
    paths.chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = PersistentClient(path=str(paths.chroma_path))
    collection = chroma_client.get_or_create_collection(paths.collection)

    sourcefile = paths.chunks_path

    with sourcefile.open("r") as f:
        for line in f:
            metadata, document = json.loads(line)
            print(f"Storing {metadata['title']}")
            metadata["date_num"] = date_num(metadata.get("date"))

            embeddings = get_embedding(document)
            # upsert so re-running embed refreshes existing chunks (and
            # backfills date_num on collections from before 2.3).
            collection.upsert(
                ids=f"{metadata['title']}_part_{metadata['part']}",
                embeddings=embeddings,
                documents=document,
                metadatas=metadata,
            )
