# Agentic RAG Assistant 🤖

A full-featured RAG (Retrieval-Augmented Generation) agent with a persistent knowledge base, a web interface for interaction and management, and multilingual capabilities. This agent uses a sophisticated stack including Streamlit, LlamaIndex, Ollama, and a PostgreSQL/pgvector database.

Its initial knowledge comes from the LlamaIndex blog post ["RAG is Dead, Long Live Agentic Retrieval"](https://www.llamaindex.ai/blog/rag-is-dead-long-live-agentic-retrieval), but its knowledge base is fully extensible via a user-friendly interface.

---

## ✨ Features

- **Interactive Chat Interface**: A web-based chat powered by Streamlit.
- **Persistent Knowledge Base**: Uses PostgreSQL with the `pgvector` extension to store and retrieve document embeddings, ensuring data persists between sessions.
- **Dynamic Knowledge Management**:
    - **Add Files**: Upload new `.txt` or `.md` documents directly through the UI.
    - **Delete Files**: Remove documents from the knowledge base.
    - **List Files**: View all documents currently indexed.
- **Source Citations**: Every response includes a list of the source documents used to generate the answer, providing transparency and trust.
- **Multilingual Responses**: Configured to understand user queries in multiple languages and respond in a specific target language (Bulgarian by default).
- **Local First**: Runs entirely on your local machine using Ollama, ensuring data privacy and no API costs.
- **Modular & Clean Code**: A well-organized codebase with separate modules for database connection, ingestion, and the main application logic.

---

## 🛠️ Technology Stack

- **Backend/RAG Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **LLM**: Local models via [Ollama](https://ollama.com/) (configured for `llama3`)
- **Web Interface**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [PostgreSQL](https://www.postgresql.org/) with [pgvector](https://github.com/pgvector/pgvector)
- **Deployment**: [Docker](https://www.docker.com/) (for the database)

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- **Python 3.10+**
- **Docker** and **Docker Compose**
- **Ollama** installed and running. You can download it from [ollama.com](https://ollama.com/).

### 2. Setup

**A. Clone the Repository**
```bash
git clone https://github.com/akaraangov/RagAgent.git
cd RagAgent

B. Pull the Ollama Model
Make sure the llama3 model is available locally.
ollama pull llama3

C. Start the PostgreSQL Database

D. Create and Activate a Virtual Environment
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

E. Install Dependencies
pip install -r requirements.txt


F. Configure Environment Variables
Create a .env file by copying the template.
# For Windows
copy .env.example .env

# For macOS/Linux
cp .env.example .env

DB_USER="admin"
DB_PASSWORD="password"
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="rag_db"

3. Ingest Initial Data
Before running the app, you need to populate the database with some initial knowledge.
Option A: Ingest from the included data directory.
Place your .txt or .md files into the data/ directory. Then run:
python ingest.py

Option B: Ingest from a custom directory.
python ingest.py /path/to/your/documents

4. Run the Streamlit App
Now you are ready to start the agent!
streamlit run app.py

Your browser should automatically open to the application's URL (usually http://localhost:8501).
🔧 How to Use the Agent
Chat: Type your questions in the input box at the bottom. The agent will respond in Bulgarian with citations.
Add Documents: Use the sidebar to upload new text files. Click "Add File" to index them.
Manage Documents: Select a file from the dropdown in the sidebar and click "Delete File" to remove it from the agent's knowledge.

📂 Project Structure
.
├── data/                  # Directory for user-provided documents
│   └── .gitkeep
├── venv/                  # Python virtual environment (ignored by Git)
├── .env                   # Environment variables (ignored by Git)
├── .gitignore             # Git ignore file
├── app.py                 # Main Streamlit application
├── db_utils.py            # Database connection logic
├── docker-compose.yml     # Docker configuration for PostgreSQL
├── ingest.py              # Script for data ingestion
├── README.md              # This file
└── requirements.txt       # Python dependencies

