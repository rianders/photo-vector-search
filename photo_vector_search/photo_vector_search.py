import logging
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from ollama import generate
import chromadb
import torch
import clip
import base64
from ollama import Client

class PhotoVectorStore:
    def __init__(self, model_name='llava-phi3:latest', persist_directory='./chroma_db'):
        self.model_name = model_name
        self.chroma_client = chromadb.PersistentClient(path=str(Path(persist_directory).resolve()))
        self.collection = self.chroma_client.get_or_create_collection(
            name="photo_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize CLIP-L model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)

    def _get_image_embedding(self, image_path):
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image).cpu().numpy()[0]
        return embedding.tolist()

    def _get_text_embedding(self, text):
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.encode_text(text_tokens).cpu().numpy()[0]
        return embedding.tolist()
    def _generate_description_with_ollama(self, photo_path, custom_prompt=None):
        prompt = custom_prompt or "Describe this image in detail."
        self.logger.debug(f"Generating description for {photo_path} with prompt: {prompt}")

        try:
            # Open and preprocess the image
            with Image.open(photo_path) as img:
                # Ensure the image is in RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize the image if necessary
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024))
                # Save the image to a BytesIO object
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()

            # Encode the image in base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Build the payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64]
            }

            # Create the Ollama client
            client = Client(base_url="http://localhost:11434")  # Adjust the base_url if necessary

            # Send the request and collect the response
            response_text = ''
            for response in client.generate(payload):
                response_text += response.get('response', '')
                # Optionally, you can log the response or handle partial outputs

            description = response_text.strip()
            self.logger.debug(f"Generated description: {description}")
            return description
        except Exception as e:
            self.logger.error(f"Ollama generate error: {str(e)}", exc_info=True)
            return ''



    def _preprocess_image(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()  # Return raw bytes

    def add_or_update_photo(self, photo_path, custom_prompt=None, aspect_name="default"):
        photo_path = Path(photo_path)
        self.logger.info(f"Processing image: {photo_path}")

        try:
            self.logger.debug(f"Generating embedding for {photo_path}")
            embedding = self._get_image_embedding(photo_path)
            self.logger.debug(f"Generated embedding (length: {len(embedding)})")
        except Exception as e:
            self.logger.error(f"Error generating embedding for {photo_path}: {str(e)}", exc_info=True)
            return False, f"Error generating embedding: {str(e)}"

        try:
            # Generate description using Ollama and llava-phi3
            description = self._generate_description_with_ollama(photo_path, custom_prompt)
            self.logger.debug(f"Generated description: {description}")
        except Exception as e:
            self.logger.error(f"Error generating description for {photo_path}: {str(e)}", exc_info=True)
            return False, f"Error generating description: {str(e)}"

        # Store in the database
        try:
            self.logger.debug(f"Checking for existing entries for {photo_path} and aspect '{aspect_name}'")
            existing_entries = self.collection.get(
                where={
                    "$and": [
                        {"photo_path": str(photo_path)},
                        {"aspect_name": aspect_name}
                    ]
                },
                include=["metadatas", "documents"]
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

    def search(self, query_image=None, query_text=None, aspect_name=None, k=5):
        self.logger.info(f"Searching with aspect: {aspect_name}, k: {k}")
        if query_image:
            self.logger.debug(f"Processing query image: {query_image}")
            embedding = self._get_image_embedding(query_image)
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
    def list_available_models(cls):
        from ollama import models
        try:
            model_list = models()
            return [model['name'] for model in model_list]
        except Exception as e:
            logging.getLogger(__name__).error(f"Error listing models: {str(e)}", exc_info=True)
            return []
