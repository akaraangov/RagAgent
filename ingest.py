import os
import argparse
import logging
import sys

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from db_utils import get_vector_store # Your database connection utility

# --- Global Configuration ---
# Configure the embedding model and node parser. These are used for processing documents.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

# Setup logging to see the progress
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def ingest_data(data_path: str):
    """
    Scans a directory for documents, loads them, processes them into nodes,
    creates embeddings, and stores them in your PostgreSQL vector database.

    Args:
        data_path (str): The path to the directory containing the files to ingest.
    """
    if not os.path.isdir(data_path):
        print(f"❌ Error: The provided path '{data_path}' is not a valid directory.")
        return

    print(f"📁 Ingesting documents from directory: {data_path}")

    # SimpleDirectoryReader will load all supported files in the directory.
    # It automatically handles .txt, .pdf, .md, and more.
    # It also automatically adds the 'file_name' to the document metadata,
    # which is crucial for the delete functionality in the Streamlit app.
    reader = SimpleDirectoryReader(input_dir=data_path, recursive=True)
    documents = reader.load_data()
    
    if not documents:
        print("No documents found in the specified directory. Exiting.")
        return
        
    print(f"Found {len(documents)} document(s) to ingest.")

    # Get the vector store instance from your utility function
    try:
        vector_store = get_vector_store()
    except Exception as e:
        # get_vector_store() already prints a detailed error, so we can just exit.
        print("Could not connect to the database. Please check your .env settings and ensure PostgreSQL is running.")
        return

    # Setup the storage context, telling LlamaIndex to use our PGVector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index. This will automatically:
    # 1. Parse documents into nodes (chunks).
    # 2. Create embeddings for each node using the configured model.
    # 3. Store the nodes and their embeddings in the pgvector table.
    print("Building index and storing data in PostgreSQL. This may take a few moments...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    
    print("\n✅ Successfully ingested data into the database.")
    print("You can now run the Streamlit app (`streamlit run app.py`) to chat with your documents.")


# --- Main Execution Block ---
# This allows you to run the script from the command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest local documents into a pgvector database.")
    parser.add_argument(
        "datapath", 
        nargs='?', 
        default="./data", 
        help="The path to the directory containing documents to ingest. Defaults to './data'."
    )
    args = parser.parse_args()
    
    ingest_data(args.datapath)