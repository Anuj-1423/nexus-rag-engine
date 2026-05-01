# Enterprise Knowledge Base Assistant

## Setup

1. Ensure MySQL is installed and running on localhost.
2. The app will create the `enterprise_rag` database automatically if it does not exist.
3. Set MySQL credentials using environment variables if your root user requires a password:
   - `DB_HOST`
   - `DB_USER`
   - `DB_PASSWORD`
   - `DB_NAME`

## Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run Frontend

The frontend is a simple React component. To run it, create a new React app and replace App.jsx with the provided file. Install axios.

```bash
npx create-react-app frontend
cd frontend
npm install axios
# Replace src/App.js with the provided App.jsx
npm start
```

## Next Upgrades

1. Replace FakeEmbeddings with Gemini/OpenAI embeddings
2. Add PDF parser
3. Add JWT middleware
4. Build proper React dashboard
5. Add citations UI
6. Docker deploy