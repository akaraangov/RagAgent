import os
import psycopg2
# THE LINE BELOW IS THE FIX:
# We import from .postgres, which matches the package you installed.
from llama_index.vector_stores.postgres import PGVectorStore 
from sqlalchemy import make_url
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_vector_store() -> PGVectorStore:
    """
    Connects to the PostgreSQL database and returns a PGVectorStore instance.
    """
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError("Database environment variables are not fully set.")

    # Create the connection string
    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # The PGVectorStore requires the db_name separately as well
    url = make_url(connection_string)

    # Attempt to connect to the database to ensure it's running
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.close()
    except psycopg2.OperationalError as e:
        print("="*80)
        print("🔴 DATABASE CONNECTION FAILED! 🔴")
        print("Please ensure your PostgreSQL server is running and configured correctly.")
        print(f"Error: {e}")
        print("="*80)
        raise

    # LlamaIndex will create the table if it doesn't exist
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name="agentic_rag_documents", # You can name your table
        embed_dim=384  # Matches the BAAI/bge-small-en-v1.5 model
    )
    return vector_store