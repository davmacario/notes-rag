# Agents Guide

## Project Type

Python vector retrieval service for markdown notes, exposing a `/query_rag` HTTP endpoint for MCP context injection. No LLM integration — returns retrieved chunks as formatted text for the calling MCP server to inject into LLM prompts.

## Stack

- **Runtime**: Python 3.14+
- **Framework**: FastMCP (MCP server), LlamaIndex (RAG orchestration)
- **Vector DB**: ChromaDB (local, persisted in `./.chromadb/` by default)
- **Embeddings**: `llama-index-embeddings-huggingface` with `all-MiniLM-L6-v2`
- **CLI**: argparse (in `cli.py`)
- **Cron parsing**: `cron_converter`
- **Git**: GitPython (`sub/git_manager.py`)
- **Package Manager**: uv

## Key Commands

```bash
# Development
uv run python -m notes_rag                     # Run MCP server + rebuild loop
uv run python -m notes_rag --verbose           # Enable DEBUG logging
uv run python -m notes_rag --only-rebuild      # Rebuild loop only, no MCP server
uv run python -m notes_rag --only-rebuild-once # Single rebuild, then exit

# Setup
uv sync              # Install dependencies
uv run pytest tests/ # Run tests
uv run ruff check    # Lint
uv run pyright       # Type check
```

## Environment Variables

Required:

```bash
export NOTES_REPO_URL="https://ghp_xxxxx@github.com/user/notes.git"
```

Optional (defaults shown):

```bash
export NOTES_DIRECTORY="./notes-cache"
export CHROMA_PATH="./.chromadb"
export TOP_K=5
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export SERVER_HOST="127.0.0.1"
export SERVER_PORT=8000
export SERVER_WORKERS=1
export SERVER_TIMEOUT=30
export REBUILD_CRON="0 */6 * * *"
export LOG_LEVEL="INFO"
```

## Development Workflow

- **Test-Driven Development**: Write tests first for all new features, then implement
- **Testing**: Use `pytest` for unit/integration tests
- **Linting**: `uv run ruff check`
- **Type checking**: `uv run pyright`
- **Run tests**: `uv run pytest tests/`

## Important Conventions

- **MCP context format**: Delimited text with `[source: path/to/file]` markers per chunk
- **GitHub auth**: Token embedded in URL as `https://ghp_xxxyy@github.com/...`
- **Local only**: All embedding generation happens locally
- **Retrieval-only**: No LLM calls — this service only returns documents for MCP to inject
- **ChromaDB persistence**: Stored in `.chromadb/` directory (gitignored)
- **Storage receives list of `Extractor` objects at initialization** for document ingestion
- **Extractors return `ExtractorResult` to Storage**: Each `Extractor` handles its own parsing and chunking, producing `TextNode` objects that `Storage` passes to `VectorStoreIndex.insert_nodes()` without re-parsing
- **Testing conventions**:
  - Use `pytest` as much as possible, especially built-in fixtures (e.g., `monkeypatch`, `caplog`)
  - For mocking objects, use `unittest.mock.MagicMock`
  - `tests/conftest.py` provides `setup_env` (autouse) and `mock_config` fixtures

## Coding Conventions

- Avoid inline comments when the logic is self-explanatory
- Avoid docstrings at the top of python files. The only docstrings present should be the ones for functions, classes, methods, and dataclass attributes
- Docstrings should be concise

## File Structure

```text
./src/notes_rag/           # Application source code
  __init__.py              # Package init
  __main__.py              # Entry point for `python -m notes_rag`
  cli.py                   # CLI entry point (argparse, daemon loop + server launch)
  config.py                # Config dataclass, env var loading with `from_env()`
  storage.py               # ChromaDB client, atomic swap rebuild, retrieval
  webserver.py             # FastAPI HTTP server with /query_rag endpoint
  logging_config.py        # Logging setup with StreamHandler
  extractor/
    abstract.py            # BaseExtractor (ABC) and ExtractorResult dataclass
    markdown_extractor.py  # MarkdownExtractor: git clone/pull, MarkdownNodeParser chunking
  sub/
    git_manager.py         # Git operations wrapper (clone, checkout, fetch, pull, status)
./tests/                   # Unit tests
  conftest.py              # pytest fixtures (setup_env, mock_config)
  test_storage.py
  test_webserver.py
  sub/
    test_git_manager.py
pyproject.toml             # uv/hatch project config, CLI script `notes-rag`
uv.lock                    # Dependency lockfile
```

## Common Mistakes

- **Don't commit tokens**: `NOTES_TOKEN` never committed; use env vars
- **Match embedding models**: Embedding model must be identical between indexing and querying
- **Embedding**: `Storage` uses LlamaIndex `VectorStoreIndex.insert_nodes()` to generate embeddings — never manually
- **Embedding model**: `llama-index-embeddings-huggingface` with `all-MiniLM-L6-v2`, passed to `VectorStoreIndex` via `embed_model` parameter
- **ChromaDB path**: Default is `.chromadb/`, overridable via `CHROMA_PATH`

## Daemon Workflow

**Startup**:

1. CLI parses args, loads `Config.from_env()`
2. `Storage` initialized with list of `Extractor` objects
3. `HTTPServer` initialized with `Storage`
4. `asyncio.gather()` starts: rebuild loop (cron-based) + `server.serve()`

**Rebuild loop**:

1. Wait until next cron time
2. Acquire `asyncio.Lock`
3. `storage.rebuild()` (run via `asyncio.to_thread()` to avoid blocking event loop)
4. Atomically swap `_client`, `_chroma_collection`, `_vector_store`, `_storage_context`, `_index`
5. Release lock
6. Errors: log and continue, old DB remains serving queries

**Query flow**:

1. MCP request to `/mcp` endpoint (via Streamable-HTTP)
2. `Storage.search()` retrieves top-k nodes
3. Format as delimited text:

   ```
   --- [source: file.md] ---
   chunk text
   --- [source: file2.md] ---
   another chunk
   ```

4. Return as response

**Shutdown** (SIGINT/SIGTERM):

1. Stop HTTP server (flush connections)
2. Stop rebuild loop
3. Exit
