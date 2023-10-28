# helper_functions.py
import json
from chromadb import PersistentClient
import openai

def get_embedding(text:str):
    embedding_object = openai.Embedding.create(input=text,model="text-embedding-ada-002")
    embedding_vector = embedding_object.data[0]['embedding']
    return embedding_vector

def store_grounding_embeddings(name: str):
    # Initialize the Chroma client and get the collection
    chroma_client = PersistentClient(path=f"data/{name}")
    collection = chroma_client.get_or_create_collection(name)

    # Determine the source file name based on the name parameter
    sourcefile = f'data/{name}_chunked.jsonl'

    # Read the chunks and their metadata from the source file
    with open(sourcefile, 'r') as f:
        for line in f:
            metadata, document = json.loads(line)
            print(f"Storing {metadata['title']}")

            # Get the OpenAI embedding for the document
            embeddings = get_embedding(document)
            # Store the document, its embedding, and its metadata in Chroma
            collection.add(ids=f"{metadata['title']}_part_{metadata['part']}", embeddings=embeddings, documents=document, metadatas=metadata)

        return