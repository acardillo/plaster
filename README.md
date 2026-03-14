# Plaster

Search for concert posters by image. In progress.

## Running the app

From the project root:

```bash
docker compose up --build
```

- **Postgres** (with pgvector): `localhost:5432`
- **Embedding service** (FastAPI): http://localhost:8001
  - `GET /health` — health check
  - `POST /embed` — upload an image, get a CLIP embedding

To run in the background: `docker compose up -d --build`

## Running tests

From the project root, run the embedding service tests:

```bash
cd embeddings && pip install -r requirements.txt && pytest -v
```

Or with a virtualenv:

```bash
cd embeddings
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest -v
```
