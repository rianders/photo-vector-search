
import streamlit as st
import os
from pathlib import Path
from PIL import Image
from photo_vector_search import PhotoVectorStore
from utils import open_image

# Initialize session state variables
if 'db_path' not in st.session_state:
    st.session_state.db_path = str(Path.home() / "tmp" / "my_chroma_db")
if 'image_directory' not in st.session_state:
    st.session_state.image_directory = str(Path.home() / "Documents" / "image_tests")
if 'store' not in st.session_state:
    st.session_state.store = PhotoVectorStore(persist_directory=st.session_state.db_path)
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

def load_image(image_path):
    return Image.open(image_path)

def main():
    st.set_page_config(layout="wide", page_title="Photo Vector Search")
    st.title("Photo Vector Search")

    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Choose a page", ["View Images", "Search Images", "Manage Images", "Settings"])

        if page == "Settings":
            st.session_state.db_path = st.text_input("ChromaDB Path", value=st.session_state.db_path)
            st.session_state.image_directory = st.text_input("Image Directory", value=st.session_state.image_directory)
            if st.button("Apply Settings"):
                st.session_state.store = PhotoVectorStore(persist_directory=st.session_state.db_path)
                st.success("Settings applied successfully!")
                st.rerun()

    if page == "View Images":
        view_images()
    elif page == "Search Images":
        search_images()
    elif page == "Manage Images":
        manage_images()

def view_images():
    st.header("View Images")
    
    # Get all unique image paths from the database
    all_entries = st.session_state.store.collection.get(include=["metadatas"])
    unique_paths = list(set(entry["photo_path"] for entry in all_entries["metadatas"]))

    # Display images in a grid
    cols = st.columns(4)
    for i, image_path in enumerate(unique_paths):
        with cols[i % 4]:
            img = load_image(image_path)
            st.image(img, caption=Path(image_path).name, use_column_width=True)
            
            if st.button(f"View Details", key=f"view_{i}"):
                st.session_state.selected_image = image_path
                st.rerun()

    if st.session_state.selected_image:
        show_image_details(st.session_state.selected_image)

def show_image_details(image_path):
    st.subheader(f"Details for {Path(image_path).name}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display the image
        img = load_image(image_path)
        st.image(img, caption=Path(image_path).name, use_column_width=True)
    
    with col2:
        # Get all aspects for this image
        entries = st.session_state.store.collection.get(
            where={"photo_path": str(image_path)},
            include=["metadatas"]
        )
        
        # Display aspects and descriptions
        for entry in entries["metadatas"]:
            with st.expander(f"Aspect: {entry['aspect_name']}", expanded=True):
                st.write(f"**Description:** {entry['description']}")

    if st.button("Close Details"):
        st.session_state.selected_image = None
        st.rerun()

def search_images():
    st.header("Search Images")
    
    tab1, tab2 = st.tabs(["Text Search", "Image Search"])
    
    with tab1:
        query_text = st.text_input("Enter search query:")
        if st.button("Search by Text"):
            with st.spinner("Searching..."):
                st.session_state.search_results = st.session_state.store.search(query_text=query_text, k=5)
            st.rerun()
    
    with tab2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Search by Image"):
                with st.spinner("Searching..."):
                    # Save the uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.search_results = st.session_state.store.search(query_image=temp_path, k=5)
                    
                    # Remove the temporary file
                    os.remove(temp_path)
                st.rerun()

    if st.session_state.search_results:
        display_search_results(st.session_state.search_results)

    if st.button("Clear Search Results"):
        st.session_state.search_results = None
        st.rerun()

def display_search_results(results):
    for photo_path, aspect, distance, description in results:
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                img = load_image(photo_path)
                st.image(img, caption=Path(photo_path).name, use_column_width=True)
            with col2:
                st.write(f"**Image:** {Path(photo_path).name}")
                st.write(f"**Aspect:** {aspect}")
                st.write(f"**Distance:** {distance:.4f}")
                st.write(f"**Description:** {description}")
        st.divider()

def manage_images():
    st.header("Manage Images")
    
    tab1, tab2, tab3 = st.tabs(["Add Image", "Update Image", "Delete Image"])
    
    with tab1:
        add_image()
    with tab2:
        update_image()
    with tab3:
        delete_image()

def add_image():
    st.subheader("Add New Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    aspect_name = st.text_input("Aspect Name", value="default")
    custom_prompt = st.text_area("Custom Prompt (optional)")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Add Image"):
            # Save the uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing image..."):
                success, message = st.session_state.store.add_or_update_photo(temp_path, custom_prompt, aspect_name)
            
            if success:
                st.success(message)
            else:
                st.error(message)
            
            # Remove the temporary file
            os.remove(temp_path)
            st.rerun()

def update_image():
    st.subheader("Update Image")
    
    # Get all unique image paths from the database
    all_entries = st.session_state.store.collection.get(include=["metadatas"])
    unique_paths = list(set(entry["photo_path"] for entry in all_entries["metadatas"]))
    
    selected_image = st.selectbox("Select image to update", unique_paths)
    aspect_name = st.text_input("Aspect Name", value="default")
    custom_prompt = st.text_area("Custom Prompt (optional)")
    
    if selected_image:
        st.image(load_image(selected_image), caption="Selected Image", use_column_width=True)
        if st.button("Update Image"):
            with st.spinner("Updating image..."):
                success, message = st.session_state.store.add_or_update_photo(selected_image, custom_prompt, aspect_name)
            if success:
                st.success(message)
            else:
                st.error(message)
            st.rerun()

def delete_image():
    st.subheader("Delete Image")
    
    # Get all unique image paths from the database
    all_entries = st.session_state.store.collection.get(include=["metadatas"])
    unique_paths = list(set(entry["photo_path"] for entry in all_entries["metadatas"]))
    
    selected_image = st.selectbox("Select image to delete", unique_paths)
    delete_all_aspects = st.checkbox("Delete all aspects", value=True)
    
    if selected_image:
        st.image(load_image(selected_image), caption="Selected Image", use_column_width=True)
        if st.button("Delete Image", type="primary"):
            with st.spinner("Deleting image..."):
                if delete_all_aspects:
                    success, message = st.session_state.store.delete_photo(selected_image)
                else:
                    aspect_name = st.text_input("Aspect Name to delete")
                    success, message = st.session_state.store.delete_photo(selected_image, aspect_name)
            if success:
                st.success(message)
            else:
                st.error(message)
            st.rerun()

if __name__ == "__main__":
    main()
