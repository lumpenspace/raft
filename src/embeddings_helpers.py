# helper_functions.py
import json
from openai import Embedding
from chromadb import Client

def get_embedding(text:str):
    return Embedding.create(text=text)

def store_grounding_embeddings(name: str):
    # Initialize the Chroma client and get the collection
    chroma_client = Client()
    collection = chroma_client.get_collection(name)

    # Determine the source file name based on the name parameter
    sourcefile = f'{name}_chunked.jsonl'

    # Read the chunks and their metadata from the source file
    with open(sourcefile, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            document = chunk['document']
            metadata = chunk['metadata']

            # Get the OpenAI embedding for the document
            embedding = get_embedding(document)

            # Store the document, its embedding, and its metadata in Chroma
            collection.add(ids=[f"{metadata['title']}_part_{metadata['part']}"], embeddings=[embedding], documents=[document], metadatas=[metadata])