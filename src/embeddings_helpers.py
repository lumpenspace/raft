# helper_functions.py
import json
from chromadb import PersistentClient
from openai import OpenAI

client = OpenAI()
import re
from enum import Enum

def get_embedding(text:str):
    embedding_object = client.embeddings.create(input=text,model="text-embedding-ada-002")
    embedding_vector = embedding_object.data[0]['embedding']
    return embedding_vector

def get_and_store_embedding(exchange: list, name: str, metadata: any):
    question, answer = exchange
    # Initialize the Chroma client and get the collection
    chroma_client = PersistentClient(path=f"data/{name}")
    collection = chroma_client.get_or_create_collection(name)
    id = re.sub(r'\W+', '', metadata["url"]+question[:20]).lower()

    stored_embedding = collection.get(ids=id)

    if (len(stored_embedding)):
        print("Embedding found in db")
        return stored_embedding['embeddings'][0]

    print("getting embeddings")
    # Get the OpenAI embedding for the text
    embedding = get_embedding(question)
    
    text = f"In a past interview, you answered '{question}' with:\n\n {answer}"
    # flatten the participants value into metadata
    metadata = {**metadata, **{"participants": ", ".join(metadata["participants"])}}
    # Store the text and its embedding in Chroma
    collection.add(ids=id, embeddings=embedding, documents=text, metadatas=metadata)

    return embedding



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