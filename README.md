# notes-rag

A local retrieval service for Markdown notes in a Git repository. It builds a Chroma vector index using local embeddings and exposes the `query_notes` tool over MCP Streamable HTTP; it does not call an LLM.

## Quickstart

1. Install Docker and Docker Compose, then clone this repository.
2. Create `.env` in the project directory with your repository URL:

   ```dotenv
   NOTES_REPO_URL=https://github.com/OWNER/REPOSITORY.git
   NOTES_BRANCH=main
   ```

   For a private repository, provide credentials using your Git setup. If embedding a token in the URL, protect `.env` and do not commit it.
3. Run `docker compose up -d`. The first build downloads the embedding model and indexes your notes; this can take time.
4. Connect an MCP client to `http://localhost:9099/mcp` and call `query_notes` with a `query_body`, for example `{"query": "project notes", "num_docs": 5}`. `http://localhost:9099/health` is the health endpoint.

The included [docker-compose.yaml](docker-compose.yaml) uses the published `ghcr.io/davmacario/notes-rag:latest` image and a `./data` bind mount for the repository cache and index. Create `data` before starting and ensure it is writable by container UID/GID 1000 (for example, `mkdir -p data && sudo chown 1000:1000 data` on Linux). To build locally instead, use `docker build -t ghcr.io/davmacario/notes-rag:latest .` before starting Compose.

## Configuration

Set these variables in the environment or `.env`. Paths are relative to the process working directory; Compose overrides the cache paths to use its persistent volume.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NOTES_REPO_URL` | required | Git URL of the notes repository |
| `NOTES_BRANCH` | `main` | Branch to check out |
| `NOTES_DIRECTORY` | `./notes-cache` | Local Git checkout |
| `CHROMA_PATH` | `./.chromadb` | Persistent Chroma data |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model; keep the same model for indexing and querying |
| `EMBEDDING_WORKERS` | `2` | Concurrent embedding workers |
| `SERVER_HOST` | `0.0.0.0` | MCP server bind address |
| `SERVER_PORT` | `9099` | MCP server port |
| `SERVER_WORKERS` | `1` | Reserved configuration setting |
| `SERVER_TIMEOUT` | `30` | Reserved configuration setting |
| `REBUILD_CRON` | `0 */6 * * *` | Index rebuild schedule |
| `TZ` | `Europe/Amsterdam` | Schedule timezone |
| `LOG_LEVEL` | `INFO` | Logging level |

Compose starts with `--rebuild-on-start`; without it the service waits for the first cron slot. `--verbose` enables debug logging. A rebuild keeps the previous index available while the replacement is being prepared.

## Development

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required. Run `uv sync`, then `uv run python -m notes_rag --rebuild-on-start`. Run tests with `uv run pytest` and lint with `uv run ruff check`.
