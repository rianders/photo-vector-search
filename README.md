# Photo Vector Search

Photo Vector Search is a Python-based CLI tool that allows you to index and search for images using advanced AI models. It uses Ollama's LLaVA-Phi models for image embedding generation and description, and ChromaDB for efficient vector storage and retrieval. The tool supports multiple aspects (custom descriptions) for images, enhancing the search capabilities.

## Features

- **Index Photos with Custom Aspects**: Index images from a specified directory with customizable aspects and prompts.
- **Search by Image or Text**: Search for similar photos using a query image or text description.
- **View Images**: Open images directly from the search results using your default image viewer.
- **Examine Indexed Images**: View details of indexed images, including their AI-generated descriptions and aspects.
- **Manage the Vector Store**: Clear or delete the vector store as needed.
- **Utilizes Advanced AI Models**: Leverages Ollama's LLaVA-Phi models for image understanding and text processing.
- **Efficient Storage with ChromaDB**: Uses ChromaDB for efficient vector storage and retrieval.
- **Cross-platform Compatibility**: Built using `pathlib` for seamless operation across different operating systems.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.11 or higher**
- **Poetry**: For dependency management. Install it from [Poetry's official website](https://python-poetry.org/docs/#installation).
- **Ollama**: Installed and running on your system. Install it from [Ollama's official website](https://ollama.ai).
- **Ollama Model**: The `llava-phi3:latest` model installed in Ollama.

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/yourusername/photo-vector-search.git
   cd photo-vector-search
   ```

2. **Ensure your `pyproject.toml` dependencies are up to date.** Here is an example of the dependencies section:

   ```toml
   [tool.poetry.dependencies]
   python = "^3.11"
   click = "^8.1.7"
   pillow = "^10.4.0"
   numpy = "1.24.3"
   chromadb = "^0.5.7"
   ollama = "^0.3.3"
   tqdm = "^4.66.5"
   requests = "^2.32.0"
   torch = "2.0.1"
   torchvision = "0.15.2"
   clip = {git = "https://github.com/openai/CLIP.git"}
   ftfy = "^6.1.1"
   regex = "^2023.8.8"
   ```

3. **Install the dependencies using Poetry:**

   ```bash
   poetry install
   ```

   This will create a virtual environment and install all necessary packages.

4. **Activate the virtual environment:**

   ```bash
   poetry shell
   ```

## Setting Up Ollama

1. **Start the Ollama Server:**

   ```bash
   ollama serve
   ```

2. **Install the Required Ollama Model:**

   ```bash
   ollama pull llava-phi3:latest
   ```

   Ensure the model supports image inputs.

## Usage

The tool provides several commands for managing and using the photo vector search system.

### Indexing Photos with Aspects

Index images with customizable aspects and prompts.

```bash
poetry run photo-vector-search index-photos [PHOTO_DIRECTORY] --prompt [CUSTOM_PROMPT] --aspect [ASPECT_NAME] [OPTIONS]
```

- **`PHOTO_DIRECTORY`**: The directory containing the photos to index (default: `~/Documents/image_tests`).
- **`--prompt`**: A custom prompt to guide the image description (optional).
- **`--aspect`**: A name for this particular aspect or perspective (default: `'default'`).
- **Options:**
  - `--model`: The Ollama model to use (default: `llava-phi3:latest`).
  - `--db-path`: Directory to store the ChromaDB database (default: `~/tmp/my_chroma_db`).
  - `--max-workers`: Maximum number of worker threads for parallel processing (default: `4`).
  - `--debug`: Enable debug logging.

**Example:**

```bash
poetry run photo-vector-search index-photos ~/my-photos --prompt "Describe the safety aspects in this image." --aspect safety --debug
```

### Adding a New Aspect to Existing Photos

Add new aspects (custom descriptions) to already indexed photos.

```bash
poetry run photo-vector-search add-aspect [PHOTO_PATH] --prompt [CUSTOM_PROMPT] --aspect [ASPECT_NAME] [OPTIONS]
```

- **`PHOTO_PATH`**: Path to the photo.
- **`--prompt`**: Custom prompt for the new aspect.
- **`--aspect`**: Name of the new aspect.
- **Options:**
  - `--model`: The Ollama model to use (default: `llava-phi3:latest`).
  - `--db-path`: Directory where ChromaDB is stored.
  - `--debug`: Enable debug logging.

**Example:**

```bash
poetry run photo-vector-search add-aspect ~/my-photos/image1.jpg --prompt "Describe the emotional impact of this image." --aspect emotional --debug
```

### Searching for Similar Photos by Image

Search for similar images using a query image.

```bash
poetry run photo-vector-search search-photos [QUERY_IMAGE] [OPTIONS]
```

- **`QUERY_IMAGE`**: Path to the query image.
- **Options:**
  - `--model`: The Ollama model to use (default: `llava-phi3:latest`).
  - `--db-path`: Directory where ChromaDB is stored.
  - `--k`: Number of results to return (default: `5`).
  - `--aspect`: Aspect to search by (leave empty to search all aspects).
  - `--verbose` or `-v`: Increase output verbosity (use `-v`, `-vv`, or `-vvv`).
  - `--view`: Open images for viewing.
  - `--debug`: Enable debug logging.

**Example:**

```bash
poetry run photo-vector-search search-photos ~/query-image.jpg --aspect safety --k 5 -v --view --debug
```

### Searching for Photos by Text

Search for images using a text query.

```bash
poetry run photo-vector-search search-photos-by-text [QUERY_TEXT] [OPTIONS]
```

- **`QUERY_TEXT`**: The text query to search for.
- **Options:** Same as for searching by image.

**Example:**

```bash
poetry run photo-vector-search search-photos-by-text "a serene landscape with vibrant colors" --aspect aesthetic --k 3 -v --view --debug
```

### Examining a Single Image

View details of an indexed image, including aspects and descriptions.

```bash
poetry run photo-vector-search examine-image [IMAGE_PATH] [OPTIONS]
```

- **`IMAGE_PATH`**: Path to the image to examine.
- **Options:**
  - `--db-path`: Directory where ChromaDB is stored.
  - `--debug`: Enable debug logging.

**Example:**

```bash
poetry run photo-vector-search examine-image ~/my-photos/cat.jpg --debug
```

### Listing Available Models

List all available Ollama models.

```bash
poetry run photo-vector-search list-models
```

**Example Output:**

```
Available models:
- llava-phi3:latest
- other-model
```

### Clearing the Vector Store

Remove all entries from the vector store without deleting the database directory.

```bash
poetry run photo-vector-search clear-store [OPTIONS]
```

- **Options:**
  - `--db-path`: Directory where ChromaDB is stored.
  - `--debug`: Enable debug logging.

**Example:**

```bash
poetry run photo-vector-search clear-store --debug
```

### Deleting the Entire Vector Store

Delete the entire vector store directory.

```bash
poetry run photo-vector-search delete-store [OPTIONS]
```

- **Options:**
  - `--db-path`: Directory where ChromaDB is stored.
  - `--debug`: Enable debug logging.

**Example:**

```bash
poetry run photo-vector-search delete-store --debug
```

## Benefits of the Aspect System

1. **Multi-faceted Analysis**: Analyze and categorize images from different perspectives (e.g., safety, aesthetics, emotions).
2. **Flexible Searches**: Perform searches based on specific aspects of images.
3. **Evolving Database**: Continuously enrich your image database with new perspectives without overwriting existing information.

## Testing and Running the Application

### Setting Up the Environment

1. **Ensure Python 3.11 is Installed:**

   ```bash
   python --version
   ```

   If not, install Python 3.11.

2. **Install Poetry:**

   Follow the instructions at [Poetry's official website](https://python-poetry.org/docs/#installation).

3. **Clone the Repository and Install Dependencies:**

   ```bash
   git clone https://github.com/yourusername/photo-vector-search.git
   cd photo-vector-search
   poetry install
   ```

4. **Activate the Virtual Environment:**

   ```bash
   poetry shell
   ```

### Running the Ollama Server

1. **Start Ollama Server:**

   ```bash
   ollama serve
   ```

2. **Verify the Ollama Model:**

   Ensure `llava-phi3:latest` is installed:

   ```bash
   ollama pull llava-phi3:latest
   ollama models
   ```

### Indexing Photos

```bash
poetry run photo-vector-search index-photos /path/to/your/photo_directory --debug
```

- Replace `/path/to/your/photo_directory` with your actual photo directory path.

### Searching and Viewing Photos

- **Search by Text:**

  ```bash
  poetry run photo-vector-search search-photos-by-text 'your search query' --view --debug
  ```

- **Search by Image:**

  ```bash
  poetry run photo-vector-search search-photos /path/to/query_image.jpg --view --debug
  ```

### Examining Indexed Images

```bash
poetry run photo-vector-search examine-image /path/to/photo.jpg --debug
```

### Troubleshooting

- **Command Not Found Error:**

  If you encounter an error like `Error: No such command 'search_photos_by_text'`, remember that command names use hyphens instead of underscores. Use `search-photos-by-text` instead.

- **Example:**

  ```bash
  poetry run photo-vector-search search-photos-by-text 'your search query' --view --debug
  ```

- **Enable Debug Logging:**

  Use the `--debug` flag to get detailed logs for troubleshooting.

## Project Structure

- **`photo_vector_search/`**: Main package directory
  - `__init__.py`: Package initialization file
  - `photo_vector_search.py`: Core functionality for photo indexing and searching
  - `cli.py`: Command-line interface implementation
- **`pyproject.toml`**: Project configuration and dependencies
- **`README.md`**: This file

## Contributing

Contributions to the Photo Vector Search project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note:** Ensure that all paths used in commands are absolute paths or relative to your current directory. Replace placeholders like `/path/to/your/photo_directory` with actual paths on your system.

**Reminder:** Always keep your dependencies up to date and verify compatibility when upgrading packages.
