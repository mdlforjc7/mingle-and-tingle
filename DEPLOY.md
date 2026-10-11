# Deploy the AI Companion

## Quick options

### 1. Render (recommended for free tier)
1. Push this folder to a GitHub repo
2. Go to render.com → New → Web Service
3. Connect the repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
6. (Optional) Add env var `OPENAI_API_KEY` or `GROK_API_KEY` for real LLM

### 2. Railway
1. Push to GitHub
2. railway.app → New Project → Deploy from GitHub
3. Railway detects the Procfile automatically
4. Add API keys in the Variables tab if desired

### 3. Docker (any VPS / Fly.io / Cloud Run)
```bash
docker build -t ai-companion .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... ai-companion
```

### 4. Fly.io
```bash
fly launch
fly deploy
```

## Notes
- Default memory backend is SQLite (file-based). On free hosts the filesystem is ephemeral — sessions and memories reset on restart unless you attach a persistent volume.
- For production memory, switch to a hosted vector DB (Pinecone, Qdrant Cloud, or Postgres + pgvector) and set the backend accordingly.
- Set `OPENAI_API_KEY` or `GROK_API_KEY` to use a real LLM instead of the offline mock.
- The app listens on `$PORT` (defaults to 8000 in Docker).

## Local test before deploy
```bash
pip install -r requirements.txt
PYTHONPATH=. uvicorn api:app --host 0.0.0.0 --port 8000
```
