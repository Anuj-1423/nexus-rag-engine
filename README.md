# Nexus RAG Engine

Nexus is a document-based knowledge assistant. It indexes uploaded files and answers questions using the relevant enterprise or personal documents.

## Requirements

- Python 3.10 or later
- MySQL
- A Google AI API key for embeddings and fallback answer generation
- A Groq API key for primary answer generation
- A Qdrant Cloud cluster and API key for persistent hosted vector storage
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
GOOGLE_EMBEDDING_API_KEY=your-google-ai-api-key-with-embedding-quota
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_MODEL=gemini-3.5-flash-lite
COHERE_API_KEY=
JWT_SECRET=replace-with-a-private-random-secret
QDRANT_URL=https://your-cluster-id.region.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
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

## Deploy on Vercel

Set the Vercel project **Root Directory** to the repository root (`.`), not `backend`. Vercel detects the FastAPI app through the root `app.py`; the root `requirements.txt` includes the backend dependencies. Leave the Build Command at its default.

Configure these Vercel environment variables:

- `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, and `DB_PORT`
- `GOOGLE_API_KEY`
- `GOOGLE_EMBEDDING_API_KEY` (optional; dedicated Gemini embedding key, falls back to `GOOGLE_API_KEY`)
- `GROQ_API_KEY` (primary text generation)
- `GROQ_MODEL` (optional; defaults to `openai/gpt-oss-120b`)
- `GEMINI_MODEL` (optional; defaults to `gemini-3.5-flash-lite`)
- `JWT_SECRET` (use a private random value)
- `QDRANT_URL` and `QDRANT_API_KEY`
- `COHERE_API_KEY` (optional)

Create a Qdrant Cloud free-tier cluster and add its cluster URL and API key in Vercel before deploying. The backend calls Qdrant Cloud's REST API directly, avoiding the larger Qdrant Python client and its gRPC/NumPy dependencies. Local development uses a SQLite-backed vector store when Qdrant Cloud variables are not configured; Vercel deployments require Qdrant Cloud because serverless local files are not durable. A lightweight pure-Python MySQL driver connects to Aiven. Text generation uses Groq first and falls back to Gemini 3.5 Flash-Lite; Google AI remains required for embeddings. The Vercel `GOOGLE_EMBEDDING_API_KEY` (or fallback `GOOGLE_API_KEY`) must have Gemini embedding access and available quota; local model fallback is disabled in serverless deployments.

If startup logs show MySQL connection failures, check that the Aiven service is running, the host and port match its connection details, and its IP allowlist permits connections from Vercel. The app can now start and serve its health check and frontend during a database outage, but database-backed features still require a working MySQL connection.

Existing local Chroma or embedded-Qdrant indexes are not copied to Qdrant Cloud or the new local SQLite store automatically; re-upload or migrate those documents. Profile photos written by the current profile endpoint use temporary serverless storage on Vercel and are not durable across function instances.

## Supported uploads

PDF, DOCX, TXT, and CSV documents are supported.
