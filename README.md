# Photo Vector Search

Photo Vector Search is a Python-based CLI tool that allows you to index and search for similar images using advanced AI models. It uses Ollama for image embedding generation and ChromaDB for efficient vector storage and retrieval.

## Features

- Index photos from a specified directory
- Search for similar photos using a query image
- View similar images using your default image viewer
- Clear or delete the vector store
- Utilizes Ollama's AI models for image embedding generation
- Efficient vector storage and retrieval using ChromaDB
- Cross-platform compatibility using pathlib

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Poetry (for dependency management)
- Ollama installed and running on your system

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/photo-vector-search.git
   cd photo-vector-search
   ```

2. Install the dependencies using Poetry:
   ```
   poetry install
   ```

## Usage

The tool provides several commands for managing and using the photo vector search system.

### Indexing Photos

To index photos from a directory:

```
poetry run photo-vector-search index-photos [PHOTO_DIRECTORY] --model [MODEL_NAME] --db-path [DB_PATH] [--update]
```

- `PHOTO_DIRECTORY`: The directory containing the photos to index (default: ~/Documents/image_tests)
- `MODEL_NAME`: The Ollama model to use for embedding generation (default: llava-phi3)
- `DB_PATH`: The directory to store the ChromaDB database (default: ~/tmp/my_chroma_db)
- `--update`: Update existing entries instead of skipping them

Example:
```
poetry run photo-vector-search index-photos ~/my-photos --model llava-phi3 --db-path ~/my-vector-db --update
```

### Searching for Similar Photos

To search for photos similar to a query image:

```
poetry run photo-vector-search search-photos [QUERY_IMAGE] --model [MODEL_NAME] --db-path [DB_PATH] --k [NUM_RESULTS] [--verbose] [--view]
```

- `QUERY_IMAGE`: Path to the query image
- `MODEL_NAME`: The Ollama model to use (should be the same as used for indexing)
- `DB_PATH`: The directory where the ChromaDB database is stored
- `NUM_RESULTS`: Number of similar images to return (default: 5)
- `--verbose`: Enable verbose output with image descriptions
- `--view`: Open images for viewing using the default image viewer

Example:
```
poetry run photo-vector-search search-photos ~/query-image.jpg --model llava-phi3 --db-path ~/my-vector-db --k 10 --verbose --view
```

### Listing Available Models

To list the available Ollama models:

```
poetry run photo-vector-search list-models
```

### Clearing the Vector Store

To remove all entries from the vector store:

```
poetry run photo-vector-search clear-store --db-path [DB_PATH]
```

### Deleting the Entire Vector Store

To delete the entire vector store directory:

```
poetry run photo-vector-search delete-store --db-path [DB_PATH]
```

## Project Structure

- `photo_vector_search/`: Main package directory
  - `__init__.py`: Package initialization file
  - `photo_vector_search.py`: Core functionality for photo indexing and searching
  - `cli.py`: Command-line interface implementation
  - `utils.py`: Utility functions (e.g., opening images)
- `tests/`: Directory for test files
- `pyproject.toml`: Project configuration and dependencies
- `README.md`: This file

## Contributing

Contributions to the Photo Vector Search project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.