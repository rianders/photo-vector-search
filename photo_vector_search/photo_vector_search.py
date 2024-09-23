import base64
from pathlib import Path
import chromadb
import ollama

class PhotoVectorStore:
    def __init__(self, model_name='llava-phi3:latest', persist_directory='./chroma_db', ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_client = ollama.Client(host=ollama_host)
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

        return description

    def _get_image_embedding_and_description(self, image_base64):
        prompt = "Describe this image in detail, including any text content, colors, objects, and overall composition:"
        
        response = self.ollama_client.generate(model=self.model_name, prompt=prompt, images=[image_base64])
        description = response['response']  # Changed from response.response to response['response']

        # Get embedding for the description
        embedding = self._get_text_embedding(description)

        return embedding, description

    def _get_text_embedding(self, text):
        response = self.ollama_client.embeddings(model=self.model_name, prompt=text)
        return response['embedding']  # Changed from response.embeddings to response['embedding']


    def search(self, query_image, k=5):
        query_image_path = Path(query_image)
        with open(query_image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        query_embedding, _ = self._get_image_embedding_and_description(image_base64)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "distances"]
        )

        return [(Path(result['metadata']['path']), distance) 
                for result, distance in zip(results['metadatas'][0], results['distances'][0])]

    def search_verbose(self, query_image, k=5):
        query_image_path = Path(query_image)
        with open(query_image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        query_embedding, _ = self._get_image_embedding_and_description(image_base64)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        return [{"path": Path(metadata['path']),
                 "distance": distance,
                 "description": document}
                for metadata, distance, document in zip(results['metadatas'][0], 
                                                        results['distances'][0],
                                                        results['documents'][0])]

    def search_by_text(self, query_text, k=5):
        query_embedding = self._get_text_embedding(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        return [{"path": Path(metadata['path']),
                 "distance": distance,
                 "description": document}
                for metadata, distance, document in zip(results['metadatas'][0], 
                                                        results['distances'][0],
                                                        results['documents'][0])]

    def clear_store(self):
        """Clear all entries from the vector store."""
        self.collection.delete(where={})

    @classmethod
    def list_available_models(cls, ollama_host="http://localhost:11434"):
        client = ollama.Client(host=ollama_host)
        models = client.list()
        return [model['name'] for model in models['models']]