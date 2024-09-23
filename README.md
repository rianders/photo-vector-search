# Photo Vector Search

Photo Vector Search is a Python-based CLI tool that allows you to index and search for similar images using advanced AI models. It uses Ollama's LLaVA-Phi3 model for image embedding generation and description, and ChromaDB for efficient vector storage and retrieval.

## Features

- Index photos from a specified directory
- Search for similar photos using a query image or text
- Examine individual images, including their AI-generated descriptions
- View similar images using your default image viewer
- Regenerate descriptions for individual images
- Clear or delete the vector store
- Utilizes Ollama's LLaVA-Phi3 model for image understanding and text processing
- Efficient vector storage and retrieval using ChromaDB
- Cross-platform compatibility using pathlib

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Poetry (for dependency management)
- Ollama installed and running on your system with the LLaVA-Phi3 model

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
poetry run photo-vector-search index-photos [PHOTO_DIRECTORY] --model [MODEL_NAME] --db-path [DB_PATH] [--update] [--max-workers N]
```

- `PHOTO_DIRECTORY`: The directory containing the photos to index (default: ~/Documents/image_tests)
- `MODEL_NAME`: The Ollama model to use for embedding generation (default: llava-phi3:latest)
- `DB_PATH`: The directory to store the ChromaDB database (default: ~/tmp/my_chroma_db)
- `--update`: Update existing entries instead of skipping them
- `--max-workers`: Maximum number of worker threads for parallel processing (default: 4)

Example:
```
poetry run photo-vector-search index-photos ~/my-photos --model llava-phi3:latest --db-path ~/my-vector-db --update --max-workers 8
```

### Searching for Similar Photos by Image

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
poetry run photo-vector-search search-photos ~/query-image.jpg --model llava-phi3:latest --db-path ~/my-vector-db --k 10 --verbose --view
```

### Searching for Photos by Text

To search for photos using a text query:

```
poetry run photo-vector-search search-photos-by-text [QUERY_TEXT] --model [MODEL_NAME] --db-path [DB_PATH] --k [NUM_RESULTS] [--view]
```

- `QUERY_TEXT`: The text query to search for
- Other options are the same as for image search

Example:
```
poetry run photo-vector-search search-photos-by-text "a cat sitting on a couch" --model llava-phi3:latest --db-path ~/my-vector-db --k 5 --view
```

### Examining a Single Image

To examine the details of a single indexed image:

```
poetry run photo-vector-search examine-image [IMAGE_PATH] --model [MODEL_NAME] --db-path [DB_PATH] [--view] [--regenerate]
```

- `IMAGE_PATH`: Path to the image to examine
- `--regenerate`: Regenerate the image description

Example:
```
poetry run photo-vector-search examine-image ~/my-photos/cat.jpg --model llava-phi3:latest --db-path ~/my-vector-db --view --regenerate
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