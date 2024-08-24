import json
import re
from chromadb import PersistentClient
from openai import OpenAI

client = OpenAI()


def get_embedding(text: str):
    embedding_object = client.embeddings.create(
        input=text,  # Remove the list wrapping
        model="text-embedding-ada-002"
    )
    embedding_vector = embedding_object.data[0].embedding
    return embedding_vector


def get_and_store_embedding(exchange, name, metadata):
    print("Metadata:", metadata)  # Add this line for debugging
    if isinstance(exchange, dict):
        question = exchange.get("question", "") or exchange.get("human", "")
    else:
        question = str(exchange)  # Ensure exchange is converted to a string
    # Use a default value if 'url' is not in metadata
    url = metadata.get("url", "")
    id = re.sub(r"\W+", "", f"{url}{question[:20]}").lower()

    chroma_client = PersistentClient(path=f"data/{name}")
    collection = chroma_client.get_or_create_collection(name)

    stored_embedding = collection.get(ids=id).get("embeddings")

    if stored_embedding and len(stored_embedding):
        print("Embedding found in db")
        return stored_embedding[0]

    print("getting embeddings")
    # Get the OpenAI embedding for the text
    embedding = get_embedding(question)

    # Prepare metadata
    if metadata:
        if "participants" in metadata:
            metadata = {**metadata, **{"participants": ", ".join(metadata["participants"].values())}}
    else:
        # If metadata is empty, create a minimal metadata dict
        metadata = {"source": "user_query"}

    # Store the question and its embedding in Chroma
    collection.add(ids=id, embeddings=embedding, documents=question, metadatas=metadata)

    return embedding


def store_grounding_embeddings(name: str):
    chroma_client = PersistentClient(path=f"data/{name}")
    collection = chroma_client.get_or_create_collection(name)

    sourcefile = f"data/{name}_chunked.jsonl"

    with open(sourcefile, "r") as f:
        for line in f:
            metadata, document = json.loads(line)
            print(f"Storing {metadata['title']}")

            # Get the OpenAI embedding for the document
            embeddings = get_embedding(document)
            # Store the document, its embedding, and its metadata in Chroma
            collection.add(
                ids=f"{metadata['title']}_part_{metadata['part']}",
                embeddings=embeddings,
                documents=document,
                metadatas=metadata,
            )

        return