import click
from pathlib import Path
from .photo_vector_search import PhotoVectorStore
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
def index_photos(photo_directory, model, db_path, prompt, aspect, max_workers):
    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    
    image_files = list(photo_directory.rglob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]

    def process_image(file_path):
        success, message = store.add_or_update_photo(file_path, custom_prompt=prompt, aspect_name=aspect)
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

    click.echo(f"\nIndexing complete:")
    click.echo(f"Successfully processed: {successful_count} images")
    click.echo(f"Errors encountered: {error_count} images")

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
@click.option('--aspect', default='default', help='Aspect to search by')
@click.option('--verbose', is_flag=True, help='Display detailed information including descriptions')
@click.option('--view', is_flag=True, help='Open images for viewing')
def search_photos(query_image, model, db_path, k, aspect, verbose, view):
    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    results = store.search(query_image=query_image, aspect_name=aspect, k=k)

    for photo_path, distance, description in results:
        click.echo(f"{photo_path}: Distance {distance}")
        if verbose:
            click.echo(f"Description ({aspect}): {description}")
        if view:
            click.launch(str(photo_path))
        click.echo()

@cli.command()
@click.argument('query_text')
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--aspect', default='default', help='Aspect to search by')
@click.option('--verbose', is_flag=True, help='Display detailed information including descriptions')
@click.option('--view', is_flag=True, help='Open images for viewing')
def search_photos_by_text(query_text, model, db_path, k, aspect, verbose, view):
    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    results = store.search(query_text=query_text, aspect_name=aspect, k=k)

    for photo_path, distance, description in results:
        click.echo(f"{photo_path}: Distance {distance}")
        if verbose:
            click.echo(f"Description ({aspect}): {description}")
        if view:
            click.launch(str(photo_path))
        click.echo()

@cli.command()
def list_models():
    models = PhotoVectorStore.list_available_models()
    click.echo("Available models:")
    for model in models:
        click.echo(f"- {model}")

if __name__ == '__main__':
    cli()