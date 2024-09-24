import logging
import json
import requests
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import uuid
import chromadb

class PhotoVectorStore:
    def __init__(self, model_name='llava-phi3:latest', persist_directory='./chroma_db', ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.chroma_client = chromadb.PersistentClient(path=str(Path(persist_directory).resolve()))
        self.collection = self.chroma_client.get_or_create_collection(
            name="photo_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def add_or_update_photo(self, photo_path, custom_prompt=None, aspect_name="default"):
        photo_path = Path(photo_path)
        self.logger.info(f"Processing image: {photo_path}")
        
        try:
            with Image.open(photo_path) as img:
                img_base64 = self._preprocess_image(img)
            self.logger.debug(f"Image preprocessed successfully: {photo_path}")
        except Exception as e:
            self.logger.error(f"Error opening image {photo_path}: {str(e)}")
            return False, f"Failed to open image: {str(e)}"

        try:
            self.logger.debug(f"Generating embedding and description for {photo_path}")
            embedding, description = self._get_embedding_and_description(img_base64, custom_prompt)
            self.logger.debug(f"Generated embedding (length: {len(embedding)}) and description (length: {len(description)})")
        except requests.RequestException as e:
            self.logger.error(f"API request failed for {photo_path}: {str(e)}")
            return False, f"API request failed: {str(e)}"
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error for {photo_path}: {str(e)}")
            return False, f"JSON parsing error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error processing {photo_path}: {str(e)}", exc_info=True)
            return False, f"Unexpected error: {str(e)}"

        try:
            self.logger.debug(f"Checking for existing entries for {photo_path} and aspect '{aspect_name}'")
            existing_entries = self.collection.get(
                where={
                    "$and": [
                        {"photo_path": str(photo_path)},
                        {"aspect_name": aspect_name}
                    ]
                },
                include=["metadatas", "documents", "ids"]
            )

            entry_id = f"{str(photo_path)}_{aspect_name}"
            metadata = {
                "photo_path": str(photo_path),
                "aspect_name": aspect_name,
                "description": description
            }

            if existing_entries['ids']:
                self.logger.debug(f"Updating existing entry for {photo_path} and aspect '{aspect_name}'")
                self.collection.update(
                    ids=[entry_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[description]
                )
                return True, f"Updated existing entry for {photo_path.name} with aspect '{aspect_name}'"
            else:
                self.logger.debug(f"Adding new entry for {photo_path} and aspect '{aspect_name}'")
                self.collection.add(
                    ids=[entry_id],
                    embeddings=[embedding],
                    documents=[description],
                    metadatas=[metadata]
                )
                return True, f"Added new entry for {photo_path.name} with aspect '{aspect_name}'"
        except Exception as e:
            self.logger.error(f"Error updating ChromaDB for {photo_path}: {str(e)}", exc_info=True)
            return False, f"Database update error: {str(e)}"

    def _get_embedding_and_description(self, image_base64, custom_prompt=None):
        url = f"{self.ollama_host}/api/generate"
        prompt = custom_prompt or "Describe this image in detail:"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "image": image_base64,
            "stream": False
        }
        headers = {'Content-Type': 'application/json'}
        
        self.logger.debug(f"Sending request to Ollama API: {url}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        self.logger.debug(f"Received response from Ollama API: {response.status_code}")
        result = response.json()
        
        description = result.get('response', '').strip()
        self.logger.debug(f"Extracted description (length: {len(description)})")
        
        embedding = self._get_text_embedding(description)
        self.logger.debug(f"Generated embedding (length: {len(embedding)})")
        
        return embedding, description

    def _get_text_embedding(self, text):
        url = f"{self.ollama_host}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        headers = {'Content-Type': 'application/json'}
        self.logger.debug(f"Sending embedding request to Ollama API: {url}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        embedding = result.get('embedding', [])
        self.logger.debug(f"Received embedding (length: {len(embedding)})")
        return embedding

    @staticmethod
    def _preprocess_image(img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def search(self, query_image=None, query_text=None, aspect_name=None, k=5):
        self.logger.info(f"Searching with aspect: {aspect_name}, k: {k}")
        if query_image:
            self.logger.debug(f"Processing query image: {query_image}")
            with Image.open(query_image) as img:
                img_base64 = self._preprocess_image(img)
            embedding, _ = self._get_embedding_and_description(img_base64)
        elif query_text:
            self.logger.debug(f"Processing query text: {query_text}")
            embedding = self._get_text_embedding(query_text)
        else:
            raise ValueError("Either query_image or query_text must be provided")

        self.logger.debug("Querying ChromaDB")
        query_params = {
            "query_embeddings": [embedding],
            "n_results": k,
            "include": ["metadatas", "distances"]
        }
        if aspect_name is not None:
            query_params["where"] = {"aspect_name": aspect_name}

        results = self.collection.query(**query_params)

        self.logger.debug(f"Raw results from ChromaDB: {results}")

        formatted_results = []
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            self.logger.debug(f"Processing result metadata: {metadata}")
            try:
                photo_path = Path(metadata.get("photo_path", "Unknown"))
                aspect = metadata.get("aspect_name", "Unknown")
                description = metadata.get("description", "No description")
                formatted_results.append((photo_path, aspect, distance, description))
            except Exception as e:
                self.logger.error(f"Error processing search result: {str(e)}")
                self.logger.error(f"Problematic metadata: {metadata}")

        self.logger.debug(f"Found {len(formatted_results)} results")
        return formatted_results

    def delete_photo(self, photo_path, aspect_name=None):
        photo_path = str(photo_path)
        try:
            if aspect_name:
                entry_id = f"{photo_path}_{aspect_name}"
                self.collection.delete(ids=[entry_id])
                return True, f"Deleted aspect '{aspect_name}' for photo '{photo_path}'."
            else:
                # Delete all aspects for this photo
                entries = self.collection.get(
                    where={"photo_path": photo_path},
                    include=["ids"]
                )
                if entries['ids']:
                    self.collection.delete(ids=entries['ids'])
                    return True, f"Deleted all aspects for photo '{photo_path}'."
                else:
                    return False, f"No entries found for photo '{photo_path}'."
        except Exception as e:
            self.logger.error(f"Error deleting photo {photo_path}: {str(e)}", exc_info=True)
            return False, f"Error deleting photo: {str(e)}"

    @classmethod
    def list_available_models(cls, ollama_host="http://localhost:11434"):
        url = f"{ollama_host}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        models = [model['name'] for model in response.json()['models']]
        logging.getLogger(__name__).debug(f"Available models: {models}")
        return models
