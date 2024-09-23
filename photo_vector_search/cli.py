import click
from pathlib import Path
from .photo_vector_search import PhotoVectorStore
import time
import numpy as np
from tqdm import tqdm
import shutil
from .utils import open_image

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
@click.option('--update', is_flag=True, help='Update existing entries instead of skipping')
def index_photos(photo_directory, model, db_path, update):
    available_models = PhotoVectorStore.list_available_models()
    if model not in available_models:
        click.echo(f"Error: Model '{model}' is not available. Available models are: {', '.join(available_models)}")
        return

    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    total_time = 0
    indexed_count = 0
    error_count = 0

    image_files = list(photo_directory.rglob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]

    with tqdm(total=len(image_files), desc="Indexing Progress", unit="image") as pbar:
        for file_path in image_files:
            start_time = time.time()
            try:
                if update:
                    store.add_photo(file_path)  # This will update existing entries
                else:
                    # Check if the image already exists
                    existing = store.collection.get(ids=[str(file_path)])
                    if not existing['ids']:
                        store.add_photo(file_path)
                    else:
                        tqdm.write(f"Skipping existing image: {file_path}")
                        pbar.update(1)
                        continue
                
                end_time = time.time()
                processing_time = end_time - start_time
                total_time += processing_time
                indexed_count += 1
                pbar.set_postfix({"Last": f"{processing_time:.2f}s", "Avg": f"{total_time/indexed_count:.2f}s"})
            except Exception as e:
                error_count += 1
                tqdm.write(f"Error processing {file_path}: {str(e)}")
            finally:
                pbar.update(1)

    avg_time = total_time / indexed_count if indexed_count > 0 else 0
    click.echo(f"\nIndexed {indexed_count} photos in total")
    click.echo(f"Encountered errors with {error_count} photos")
    click.echo(f"Total time: {total_time:.2f} seconds")
    click.echo(f"Average time per photo: {avg_time:.2f} seconds")

@cli.command()
@click.argument('query_image', type=click.Path(exists=True, path_type=Path))
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--view', is_flag=True, help='Open images for viewing')
def search_photos(query_image, model, db_path, k, verbose, view):
    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    with tqdm(total=1, desc="Searching", unit="query") as pbar:
        if verbose:
            results = store.search_verbose(query_image, k)
            for i, result in enumerate(results, 1):
                click.echo(f"\nResult {i}:")
                click.echo(f"Path: {result['path']}")
                click.echo(f"Distance: {result['distance']}")
                click.echo(f"Description: {result['description']}")
                if view:
                    click.echo("Opening image...")
                    open_image(result['path'])
                    if i < len(results):
                        if not click.confirm('View next image?'):
                            break
        else:
            results = store.search(query_image, k)
            for i, (photo_path, distance) in enumerate(results, 1):
                click.echo(f"Result {i}: {photo_path} (Distance: {distance})")
                if view:
                    click.echo("Opening image...")
                    open_image(photo_path)
                    if i < len(results):
                        if not click.confirm('View next image?'):
                            break
        
        if view:
            click.echo("Opening query image...")
            open_image(query_image)
        
        pbar.update(1)

@cli.command()
@click.argument('query_text')
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--view', is_flag=True, help='Open images for viewing')
def search_photos_by_text(query_text, model, db_path, k, view):
    click.echo(f"Searching for: '{query_text}'")
    click.echo(f"Using model: {model}")
    click.echo(f"Database path: {db_path}")
    click.echo(f"Number of results requested: {k}")

    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    
    with tqdm(total=1, desc="Searching", unit="query") as pbar:
        try:
            results = store.search_by_text(query_text, k)
            
            pbar.update(1)

            if not results:
                click.echo("No matching images found.")
                return

            click.echo(f"\nFound {len(results)} results:")
            
            for i, (photo_path, distance, description) in enumerate(results, 1):
                click.echo(f"\nResult {i}:")
                click.echo(f"Path: {photo_path}")
                click.echo(f"Distance: {distance}")
                click.echo(f"Description: {description}")
                
                if view:
                    click.echo("Opening image...")
                    open_image(photo_path)
                    if i < len(results) and not click.confirm('View next image?'):
                        break

        except Exception as e:
            click.echo(f"An error occurred during the search: {str(e)}")
            click.echo("Please check the console output for more detailed error information.")


@cli.command()
def list_models():
    models = PhotoVectorStore.list_available_models()
    click.echo("Available models:")
    for model in models:
        click.echo(f"- {model}")

@cli.command()
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.confirmation_option(prompt='Are you sure you want to clear the vector store?')
def clear_store(db_path):
    """Clear all entries from the vector store."""
    store = PhotoVectorStore(persist_directory=str(db_path))
    store.clear_store()
    click.echo("Vector store cleared successfully.")

@cli.command()
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.confirmation_option(prompt='Are you sure you want to delete the entire vector store?')
def delete_store(db_path):
    """Delete the entire vector store directory."""
    try:
        shutil.rmtree(db_path)
        click.echo(f"Vector store at {db_path} has been deleted.")
    except FileNotFoundError:
        click.echo(f"Vector store not found at {db_path}.")
    except Exception as e:
        click.echo(f"An error occurred while trying to delete the vector store: {e}")

@cli.command()
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use')
@click.option('--db-path', type=click.Path(path_type=Path), default=DEFAULT_DB_PATH, help='Directory where ChromaDB is stored')
@click.option('--view', is_flag=True, help='Open the image for viewing')
def examine_image(image_path, model, db_path, view):
    """Examine a single image's description and metadata."""
    click.echo(f"Examining image: {image_path}")
    click.echo(f"Using model: {model}")
    click.echo(f"Database path: {db_path}")

    store = PhotoVectorStore(model_name=model, persist_directory=str(db_path))
    
    image_info = store.examine_image(image_path)
    
    if image_info is None:
        click.echo("Image not found in the database. Make sure you've indexed this image.")
        return

    click.echo("\nImage Information:")
    click.echo(f"Path: {image_info['path']}")
    click.echo(f"Description: {image_info['description']}")
    click.echo(f"Metadata: {image_info['metadata']}")
    
    embedding = np.array(image_info['embedding'])
    click.echo(f"Embedding shape: {embedding.shape}")
    click.echo(f"Embedding summary: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}, std={embedding.std():.4f}")

    if view:
        click.echo("\nOpening image...")
        open_image(image_path)

if __name__ == '__main__':
    cli()