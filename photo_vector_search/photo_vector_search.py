from pathlib import Path
from PIL import Image
import numpy as np
import chromadb
from chromadb.config import Settings
import base64
from io import BytesIO
import requests
import json

class PhotoVectorStore:
    def __init__(self, model_name='llava:7b', persist_directory='./chroma_db', ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.chroma_client = chromadb.PersistentClient(path=str(Path(persist_directory).resolve()))
        self.collection = self.chroma_client.get_or_create_collection(name="photo_collection")

    def add_photo(self, photo_path):
        photo_path = Path(photo_path)
        with Image.open(photo_path) as img:
            img_base64 = self._preprocess_image(img)

        embedding = self._get_embedding(img_base64)

        # Check if the image already exists in the collection
        existing = self.collection.get(ids=[str(photo_path)])
        if existing['ids']:
            # Update the existing entry
            self.collection.update(
                ids=[str(photo_path)],
                embeddings=[embedding],
                metadatas=[{"path": str(photo_path)}]
            )
        else:
            # Add a new entry
            self.collection.add(
                embeddings=[embedding],
                documents=[f"Image: {photo_path.name}"],
                metadatas=[{"path": str(photo_path)}],
                ids=[str(photo_path)]
            )

    def clear_store(self):
        """Clear all entries from the vector store."""
        self.collection.delete(where={})

    def search(self, query_image, k=5):
        with Image.open(query_image) as img:
            img_base64 = self._preprocess_image(img)

        # Generate embedding for query image
        try:
            query_embedding = self._get_embedding(img_base64)
        except Exception as e:
            print(f"Error generating embedding for query image: {str(e)}")
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Debug: Print the structure of the results
        print("Results structure:", json.dumps(results, indent=2))
        
        # Check if 'metadatas' key exists and is not empty
        if 'metadatas' not in results or not results['metadatas']:
            print("No results found or unexpected result structure.")
            return []
        
        # Safely access and process the results
        processed_results = []
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            if 'path' in metadata:
                processed_results.append((Path(metadata['path']), distance))
            else:
                print(f"Warning: 'path' not found in metadata: {metadata}")
        
        return processed_results

    def _get_embedding(self, image_base64):
        url = f"{self.ollama_host}/api/embeddings"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": "Describe this image:",
            "image": image_base64
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()['embedding']


    def search_verbose(self, query_image, k=5):
        with Image.open(query_image) as img:
            img_base64 = self._preprocess_image(img)

        # Generate embedding and description for query image
        try:
            query_embedding = self._get_embedding(img_base64)
            query_description = self._get_description(img_base64)
            print(f"Query image description: {query_description}")
        except Exception as e:
            print(f"Error processing query image: {str(e)}")
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )
        
        processed_results = []
        for metadata, document, distance in zip(results['metadatas'][0], results['documents'][0], results['distances'][0]):
            if 'path' in metadata:
                path = Path(metadata['path'])
                with Image.open(path) as img:
                    img_base64 = self._preprocess_image(img)
                    description = self._get_description(img_base64)
                processed_results.append({
                    'path': path,
                    'distance': distance,
                    'description': description,
                    'document': document
                })
            else:
                print(f"Warning: 'path' not found in metadata: {metadata}")
        
        return processed_results

    def _get_description(self, image_base64):
        url = f"{self.ollama_host}/api/generate"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": "Describe this image in detail:",
            "images": [image_base64],
            "stream": False  # Set to False to get a complete response
        })
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    full_response += json_response.get('response', '')
                    if json_response.get('done', False):
                        break
            
            return full_response.strip()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return "Error: Unable to decode API response"
        except requests.exceptions.RequestException as e:
            print(f"Error making request to Ollama API: {e}")
            return "Error: Unable to communicate with Ollama API"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "Error: Unexpected error occurred"


    def search_by_text(self, query_text, k=5):
        # First, let's check how many items are in the collection
        collection_size = self.collection.count()
        print(f"Total items in the vector store: {collection_size}")

        query_embedding = self._get_text_embedding(query_text)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection_size),  # Ensure we don't request more than what's in the collection
                include=["metadatas", "distances", "documents"],
                where={},  # Remove any filters
                where_document={},  # Remove any document filters
            )
            
            print(f"Raw query results: {json.dumps(results, indent=2)}")  # Print raw results for debugging

            if not results['ids'][0]:
                print("No results found. Trying with a lower threshold...")
                # Try again with a much lower threshold
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k, collection_size),
                    include=["metadatas", "distances", "documents"],
                    where={},
                    where_document={},
                    score_threshold=0.5  # This is a relatively low threshold, adjust as needed
                )
                print(f"Raw query results (with lower threshold): {json.dumps(results, indent=2)}")  # Print raw results for debugging

            return [
                (
                    Path(result.get('path', 'Unknown path')),  # Use 'Unknown path' if 'path' is not found
                    distance
                )
                for result, distance in zip(results['metadatas'][0], results['distances'][0])
            ]
        except Exception as e:
            print(f"Error in search_by_text: {str(e)}")
            print(f"Results structure: {results}")  # Print the structure of results
            raise  # Re-raise the exception for the calling function to handle


    def search_by_text_verbose(self, query_text, k=5):
        query_embedding = self._get_text_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )
        return [
            {
                'path': Path(result['metadata']['path']),
                'distance': distance,
                'description': document.split('Image: ')[1] if 'Image: ' in document else document
            }
            for result, distance, document in zip(results['metadatas'][0], results['distances'][0], results['documents'][0])
        ]

    def _get_text_embedding(self, text):
        url = f"{self.ollama_host}/api/embeddings"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": text
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        embedding = response.json()['embedding']
        print(f"Generated embedding for '{text}': {embedding[:5]}... (showing first 5 elements)")
        return embedding

    @staticmethod
    def _preprocess_image(img):
        # Convert image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize image if it's too large
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @classmethod
    def list_available_models(cls, ollama_host="http://localhost:11434"):
        url = f"{ollama_host}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return [model['name'] for model in response.json()['models']]