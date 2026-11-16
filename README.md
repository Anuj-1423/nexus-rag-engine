# Nexus RAG Engine

Nexus is a document-based knowledge assistant. It indexes uploaded files and answers questions using the relevant enterprise or personal documents.

## Requirements

- Python 3.10 or later
- MySQL
- A Google AI API key for embeddings and answer generation
- A Cohere API key is optional; retrieval uses a local reranking fallback when it is not configured

## Configuration

Create `backend/.env` with the settings for your environment. Keep this file private; it is ignored by Git.

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=defaultdb
DB_PORT=3306
GOOGLE_API_KEY=your-google-ai-api-key
COHERE_API_KEY=
JWT_SECRET=replace-with-a-private-random-secret
```

The application initializes its database and tables when the backend starts. Use credentials that have permission to connect to MySQL and create the configured database, or create it beforehand.

## Install and run

From the repository root, create and activate a virtual environment, then install the backend dependencies:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The backend serves the frontend as well. Open <http://127.0.0.1:8000/> for the website, or <http://127.0.0.1:8000/docs> for the API documentation.

## Supported uploads

PDF, DOCX, TXT, and CSV documents are supported.
