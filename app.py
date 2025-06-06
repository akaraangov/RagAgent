import streamlit as st
import os
import psycopg2
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from db_utils import get_vector_store

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load environment variables from .env file
load_dotenv()

st.title("🤖 Agentic RAG Assistant")
st.caption("A RAG agent powered by local documents and a PostgreSQL vector database.")

# --- Model and Embedding Configuration ---
# Use a cache to avoid reloading models on every run
@st.cache_resource
def configure_models():
    """Load and cache the LLM and embedding model."""
    print("Configuring models with Bulgarian system prompt...")
    
    # --- THE CHANGE IS HERE ---
    # We define a system prompt in Bulgarian to guide the model's responses.
    bulgarian_system_prompt = (
        "Ти си интелигентен асистент. Твоята задача е да отговаряш на въпросите на потребителя, "
        "като използваш предоставения контекст. Всички твои отговори трябва да бъдат на български език."
    )
    # Translation:
    # "You are an intelligent assistant. Your task is to answer the user's questions "
    # "using the provided context. All of your responses must be in the Bulgarian language."
    
    # We pass the system_prompt when initializing the Ollama LLM.
    Settings.llm = Ollama(
        model="mistral", 
        request_timeout=120.0,
        system_prompt=bulgarian_system_prompt
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

configure_models()

# --- Vector Store and Index Initialization ---
@st.cache_resource
def get_index():
    """Get the VectorStoreIndex from the pgvector store."""
    print("Connecting to vector store and getting index...")
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)

try:
    index = get_index()
except Exception as e:
    st.error("🔴 Failed to connect to the database and initialize the index.")
    st.error("Please ensure your PostgreSQL container is running and .env settings are correct.")
    st.exception(e)
    st.stop()


# --- Sidebar for Knowledge Base Management ---
st.sidebar.title("📚 Knowledge Base")

# --- FIX: Reliable way to fetch stored files ---
# We now use a direct psycopg2 connection, which is more stable than relying
# on an internal attribute of the VectorStore object.

@st.cache_data(ttl=10) # Cache the result for 10 seconds to improve performance
def get_stored_files():
    """
    Retrieves the list of unique file names from the vector store's metadata.
    """
    try:
        # Establish a direct connection to the database
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        with conn.cursor() as c:
            # --- THE FIX IS IN THIS SQL QUERY ---
            # We use the ->> operator to extract the file_name as text.
            # This works for both 'json' and 'jsonb' types.
            # We then check if the result is not NULL, which confirms the key exists.
            sql_query = """
                SELECT DISTINCT metadata_->>'file_name' 
                FROM data_agentic_rag_documents 
                WHERE metadata_->>'file_name' IS NOT NULL;
            """
            c.execute(sql_query)
            results = c.fetchall()
        conn.close()
        # Return a list of filenames
        return [row[0] for row in results if row and row[0]]
    except psycopg2.errors.UndefinedTable:
        # This is a special case for the very first run when the table doesn't exist yet.
        # It's not an error, just means there are no files.
        return []
    except Exception as e:
        # For all other errors, display them in the sidebar.
        st.sidebar.error(f"DB Error: {e}")
        return []


# --- File Uploader and Management ---
uploaded_file = st.sidebar.file_uploader(
    "Add a new file to the knowledge base",
    type=["txt", "md"],
    help="Upload a file to be indexed and made available to the agent."
)


if st.sidebar.button("Add File", disabled=not uploaded_file):
    with st.spinner("Indexing new file... Please wait."):
        try:
            # Read file content directly from memory
            file_content = uploaded_file.getvalue().decode("utf-8")
            
            # --- THE FIX IS ON THIS LINE ---
            # We must explicitly add the filename to the metadata dictionary.
            # This is what our get_stored_files() function looks for.
            document = Document(
                text=file_content,
                doc_id=uploaded_file.name,
                metadata={"file_name": uploaded_file.name}
            )

            # The .insert() method expects a single Document object.
            index.insert(document)
            
            st.sidebar.success(f"✅ Successfully added '{uploaded_file.name}'")
            st.cache_data.clear() # Clear the file list cache
            st.rerun() # Refresh the page to show the new file

        except Exception as e:
            st.sidebar.error(f"❌ Error adding file: {e}")
            st.exception(e) # Show full error details in the app


# --- Display and Delete Files ---
st.sidebar.markdown("---")
st.sidebar.subheader("Manage Stored Files")

stored_files = get_stored_files()

if not stored_files:
    st.sidebar.info("No files found in the knowledge base yet.")
else:
    file_to_manage = st.sidebar.selectbox(
        "Select a file to manage",
        options=stored_files
    )

    if st.sidebar.button("Delete File", type="primary"):
        with st.spinner(f"Deleting '{file_to_manage}'..."):
            try:
                # --- FIX: Use the correct deletion method ---
                # We delete using the ref_doc_id, which we set to be the file name.
                index.delete_ref_doc(file_to_manage, delete_from_docstore=True)
                
                st.sidebar.success(f"🗑️ Successfully deleted '{file_to_manage}'")
                st.cache_data.clear() # Clear the file list cache
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error deleting file. See console for details.")
                st.exception(e)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Здравейте! Готов съм да отговарям на въпроси, базирани на документите в моята база от знания."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Задайте въпроса си тук..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Мисля..."):
            query_engine = index.as_query_engine(streaming=True)
            streaming_response = query_engine.query(prompt)
            
            # First, stream the text response to the user
            full_response = ""
            for text in streaming_response.response_gen:
                full_response += text
                message_placeholder.markdown(full_response + "▌")
            
            # --- THE CITATION LOGIC STARTS HERE ---
            # After streaming is complete, process the source nodes
            citations_markdown = ""
            if streaming_response.source_nodes:
                # Use a set to store unique filenames
                source_files = set()
                for node in streaming_response.source_nodes:
                    if 'file_name' in node.metadata:
                        source_files.add(node.metadata['file_name'])
                
                # Format the citations into a markdown string
                if source_files:
                    citations_markdown += "\n\n---\n**Източници:**" # "Sources:" in Bulgarian
                    for filename in sorted(list(source_files)): # Sort for consistent order
                        citations_markdown += f"\n* `{filename}`"

            # Combine the final response with the formatted citations
            final_output = full_response + citations_markdown
            
            # Update the message placeholder with the final, complete output
            message_placeholder.markdown(final_output)
    
    # Save the complete output (response + citations) to the session state
    st.session_state.messages.append({"role": "assistant", "content": final_output})