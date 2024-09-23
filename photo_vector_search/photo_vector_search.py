import requests
import json
from pathlib import Path
import base64
import chromadb
import numpy as np

class PhotoVectorStore:
    def __init__(self, model_name='llava-phi3:latest', persist_directory='./chroma_db', ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.chroma_client = chromadb.PersistentClient(path=str(Path(persist_directory).resolve()))
        self.collection = self.chroma_client.get_or_create_collection(name="photo_collection")

    def add_photo(self, photo_path):
        photo_path = Path(photo_path)
        with open(photo_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Generate embedding and description
        embedding, description = self._get_image_embedding_and_description(image_base64)

        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[description],
            metadatas=[{"path": str(photo_path)}],
            ids=[str(photo_path)]
        )

    def search_by_text(self, query_text, k=5):
        query_embedding = self._get_text_embedding(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        return [
            (Path(result['metadata']['path']), distance, document)
            for result, distance, document in zip(results['metadatas'][0], results['distances'][0], results['documents'][0])
        ]

    def _get_image_embedding_and_description(self, image_base64):
        url = f"{self.ollama_host}/api/generate"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": "Describe this image in detail:",
            "images": [image_base64]
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        description = response.json()['response']

        # Get embedding for the description
        embedding = self._get_text_embedding(description)

        return embedding, description

    def _get_text_embedding(self, text):
        url = f"{self.ollama_host}/api/embeddings"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": text
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()['embedding']
 
    def examine_image(self, image_path):
        image_path = str(Path(image_path).resolve())
        results = self.collection.get(
            ids=[image_path],
            include=["embeddings", "documents", "metadatas"]
        )
        
        if not results['ids']:
            return None
        
        return {
            'path': results['ids'][0],
            'description': results['documents'][0],
            'embedding': results['embeddings'][0],
            'metadata': results['metadatas'][0]
        }

    def clear_store(self):
        """Clear all entries from the vector store."""
        self.collection.delete(where={})

    @classmethod
    def list_available_models(cls, ollama_host="http://localhost:11434"):
        url = f"{ollama_host}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return [model['name'] for model in response.json()['models']]