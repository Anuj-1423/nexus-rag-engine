# Nexus RAG Engine

Nexus is a document-based knowledge assistant. It indexes uploaded files and answers questions using the relevant enterprise or personal documents.

## Requirements

- Python 3.10 or later
- MySQL
- A Google AI API key for embeddings and answer generation
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

Set the Vercel project **Root Directory** to the repository root (`.`), not `backend`. The root `pyproject.toml` points Vercel to the FastAPI app, and the root `requirements.txt` includes the backend dependencies. Leave the Build Command at its default; do not add another `backend/` prefix.

Configure these Vercel environment variables:

- `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, and `DB_PORT`
- `GOOGLE_API_KEY`
- `JWT_SECRET` (use a private random value)
- `QDRANT_URL` and `QDRANT_API_KEY`
- `COHERE_API_KEY` (optional)

Create a Qdrant Cloud free-tier cluster and add its cluster URL and API key in Vercel before deploying. The backend calls Qdrant Cloud's REST API directly, avoiding the larger Qdrant Python client and its gRPC/NumPy dependencies. Local development uses a SQLite-backed vector store when Qdrant Cloud variables are not configured; Vercel deployments require Qdrant Cloud because serverless local files are not durable. A lightweight pure-Python MySQL driver connects to Aiven. Semantic chunking is optional; when its extra package is absent, the app uses recursive chunking.

Existing local Chroma or embedded-Qdrant indexes are not copied to Qdrant Cloud or the new local SQLite store automatically; re-upload or migrate those documents. Profile photos written by the current profile endpoint use temporary serverless storage on Vercel and are not durable across function instances.

## Supported uploads

PDF, DOCX, TXT, and CSV documents are supported.
