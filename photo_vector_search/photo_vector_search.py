import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from PIL import Image
import requests
import json
import uuid
from io import BytesIO
import base64
import logging


class PhotoVectorStore:
    def __init__(self, model_name='llava-phi3:latest', persist_directory='./chroma_db', ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.chroma_client = chromadb.PersistentClient(path=str(Path(persist_directory).resolve()))
        self.collection = self.chroma_client.get_or_create_collection(
            name="photo_collection",
            metadata={"hnsw:space": "cosine"}
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def add_or_update_photo(self, photo_path, custom_prompt=None, aspect_name="default"):
        photo_path = Path(photo_path)
        self.logger.info(f"Processing image: {photo_path}")
        
        try:
            with Image.open(photo_path) as img:
                img_base64 = self._preprocess_image(img)
        except Exception as e:
            self.logger.error(f"Error opening image {photo_path}: {str(e)}")
            return False, f"Failed to open image: {str(e)}"

        try:
            embedding, description = self._get_embedding_and_description(img_base64, custom_prompt)
        except RequestException as e:
            self.logger.error(f"API request failed for {photo_path}: {str(e)}")
            return False, f"API request failed: {str(e)}"
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error for {photo_path}: {str(e)}")
            return False, f"JSON parsing error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Unexpected error processing {photo_path}: {str(e)}", exc_info=True)
            return False, f"Unexpected error: {str(e)}"

        # Check if the photo already exists in the collection
        existing_entries = self.collection.get(
            where={"photo_path": str(photo_path)},
            include=["metadatas", "documents"]
        )

        if existing_entries['ids']:
            # Update existing entry
            entry_id = existing_entries['ids'][0]
            current_metadata = existing_entries['metadatas'][0]
            current_metadata[f"description_{aspect_name}"] = description
            
            self.collection.update(
                ids=[entry_id],
                metadatas=[current_metadata],
                documents=[json.dumps(current_metadata)]
            )
            return True, f"Updated existing entry for {photo_path.name} with aspect '{aspect_name}'"
        else:
            # Add new entry
            entry_id = str(uuid.uuid4())
            metadata = {
                "photo_path": str(photo_path),
                f"description_{aspect_name}": description
            }

            self.collection.add(
                ids=[entry_id],
                embeddings=[embedding],
                documents=[json.dumps(metadata)],
                metadatas=[metadata]
            )
            return True, f"Added new entry for {photo_path.name} with aspect '{aspect_name}'"

    def search(self, query_image=None, query_text=None, aspect_name="default", k=5):
        if query_image:
            with Image.open(query_image) as img:
                img_base64 = self._preprocess_image(img)
            embedding, _ = self._get_embedding_and_description(img_base64)
        elif query_text:
            embedding = self._get_text_embedding(query_text)
        else:
            raise ValueError("Either query_image or query_text must be provided")

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["metadatas", "distances"]
        )

        return [
            (
                Path(metadata["photo_path"]),
                distance,
                metadata.get(f"description_{aspect_name}", "No description for this aspect")
            )
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0])
        ]


    def _get_embedding_and_description(self, image_base64, custom_prompt=None):
        url = f"{self.ollama_host}/api/generate"
        prompt = custom_prompt or "Describe this image in detail:"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "image": image_base64,
            "stream": True  # We'll use streaming to handle the response line by line
        }
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        description = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line)
                    if 'response' in json_line:
                        description += json_line['response']
                except json.JSONDecodeError:
                    self.logger.warning(f"Couldn't parse line as JSON: {line}")
                    continue

        description = description.strip()
        
        # Get embedding for the description
        embedding = self._get_text_embedding(description)
        
        return embedding, description

    def _get_text_embedding(self, text):
        url = f"{self.ollama_host}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get('embedding', [])

    @staticmethod
    def _preprocess_image(img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @classmethod
    def list_available_models(cls, ollama_host="http://localhost:11434"):
        url = f"{ollama_host}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return [model['name'] for model in response.json()['models']]