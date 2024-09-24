import click
from pathlib import Path
from .photo_vector_search import PhotoVectorStore
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import shutil

DEFAULT_DOC_SOURCE = Path.home() / "Documents" / "image_tests"
DEFAULT_DB_PATH = Path.home() / "tmp" / "my_chroma_db"
DEFAULT_MODEL = "llava-phi3:latest"

@click.group()
def cli():
    pass

@cli.command()
@click.argument('photo_directory', type=click.Path(exists=True, path_type=Path), default=DEFAULT_DOC_SOURCE)
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory to store ChromaDB')
@click.option('--prompt', default=None, help='Custom prompt for image description')
@click.option('--aspect', default='default', help='Name of the aspect to index')
@click.option('--max-workers', default=4, help='Maximum number of worker threads')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def index_photos(photo_directory, model, db_path, prompt, aspect, max_workers, debug):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Indexing photos from {photo_directory}")
    logger.info(f"Using model: {model}")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Aspect: {aspect}")

    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    
    image_files = list(photo_directory.rglob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    logger.info(f"Found {len(image_files)} image files")

    def process_image(file_path):
        logger.debug(f"Processing {file_path}")
        success, message = store.add_or_update_photo(file_path, custom_prompt=prompt, aspect_name=aspect)
        logger.debug(f"Result for {file_path}: {message}")
        return success, message

    successful_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, file_path) for file_path in image_files]
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Indexing Progress"):
            success, message = future.result()
            if success:
                successful_count += 1
            else:
                error_count += 1
            tqdm.write(message)

    logger.info(f"\nIndexing complete:")
    logger.info(f"Successfully processed: {successful_count} images")
    logger.info(f"Errors encountered: {error_count} images")

@cli.command()
@click.argument('photo_path', type=click.Path(exists=True, path_type=Path))
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.option('--prompt', required=True, help='Custom prompt for the new aspect')
@click.option('--aspect', required=True, help='Name of the new aspect')
def add_aspect(photo_path, model, db_path, prompt, aspect):
    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    store.add_or_update_photo(photo_path, custom_prompt=prompt, aspect_name=aspect)
    click.echo(f"Added aspect '{aspect}' to {photo_path}")

@cli.command()
@click.argument('query_image', type=click.Path(exists=True, path_type=Path))
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--aspect', default=None, help='Aspect to search by (leave empty to search all aspects)')
@click.option('--verbose', is_flag=True, help='Display detailed information including descriptions')
@click.option('--view', is_flag=True, help='Open images for viewing')
def search_photos(query_image, model, db_path, k, aspect, verbose, view):
    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    results = store.search(query_image=query_image, aspect_name=aspect, k=k)

    for photo_path, aspect_name, distance, description in results:
        click.echo(f"Photo: {photo_path}")
        click.echo(f"Aspect: {aspect_name}")
        click.echo(f"Distance: {distance}")
        if verbose:
            click.echo(f"Description ({aspect_name}): {description}")
        if view:
            click.launch(str(photo_path))
        click.echo()

@cli.command()
@click.argument('query_text')
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--aspect', default=None, help='Aspect to search by (leave empty to search all aspects)')
@click.option('--verbose', count=True, help='Increase output verbosity (e.g., -v, -vv, -vvv)')
@click.option('--view', is_flag=True, help='Open images for viewing')
def search_photos_by_text(query_text, model, db_path, k, aspect, verbose, view):
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(verbose, len(log_levels) - 1)]
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    
    try:
        results = store.search(query_text=query_text, aspect_name=aspect, k=k)
        
        if not results:
            click.echo("No results found.")
            return

        for photo_path, aspect_name, distance, description in results:
            click.echo(f"Photo: {photo_path}")
            click.echo(f"Aspect: {aspect_name}")
            click.echo(f"Distance: {distance}")
            click.echo(f"Description ({aspect_name}): {description}")
            if view:
                click.launch(str(photo_path))
            click.echo()
    except Exception as e:
        logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        click.echo(f"An error occurred during search. Please check the logs for more details.")

@cli.command()
def list_models():
    models = PhotoVectorStore.list_available_models()
    click.echo("Available models:")
    for model in models:
        click.echo(f"- {model}")

@cli.command()
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
def clear_store(db_path):
    """Remove all entries from the vector store without deleting the database directory."""
    store = PhotoVectorStore(persist_directory=str(db_path))
    try:
        store.collection.delete(where={})
        click.echo(f"Cleared all entries from the vector store at '{db_path}'.")
    except Exception as e:
        click.echo(f"An error occurred while clearing the vector store: {str(e)}")

@cli.command()
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
def delete_store(db_path):
    """Delete the entire vector store directory."""
    db_path = Path(db_path)
    if db_path.exists():
        click.confirm(f"Are you sure you want to delete the entire vector store at '{db_path}'?", abort=True)
        shutil.rmtree(db_path)
        click.echo(f"Deleted the vector store at '{db_path}'.")
    else:
        click.echo(f"No vector store found at '{db_path}'.")

@cli.command()
@click.argument('photo_path', type=click.Path(exists=True, path_type=Path))
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
def examine_image(photo_path, db_path):
    """Examine the details of a single indexed image."""
    store = PhotoVectorStore(persist_directory=str(db_path))
    photo_path = str(Path(photo_path))
    entries = store.collection.get(
        where={"photo_path": photo_path},
        include=["metadatas", "documents"]
    )

    if entries['ids']:
        click.echo(f"Found entries for photo: {photo_path}")
        for metadata in entries['metadatas']:
            aspect = metadata.get("aspect_name", "Unknown")
            description = metadata.get("description", "No description")
            click.echo(f"Aspect: {aspect}")
            click.echo(f"Description: {description}")
            click.echo()
    else:
        click.echo(f"No entries found for photo: {photo_path}")

if __name__ == '__main__':
    cli()
