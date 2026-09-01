# Agents Guide

## Project Type

Python vector retrieval service for markdown notes, exposing a `query_notes` MCP tool (FastMCP, Streamable-HTTP transport) for context injection. No LLM integration — returns retrieved chunks as formatted text for the calling MCP client to inject into LLM prompts.

## Stack

- **Runtime**: Python 3.14+
- **Framework**: FastMCP (MCP server), LlamaIndex (RAG orchestration)
- **Vector DB**: ChromaDB (local, persisted in `./.chromadb/` by default)
- **Embeddings**: `llama-index-embeddings-fastembed` (ONNX, no torch) with `sentence-transformers/all-MiniLM-L6-v2`
- **CLI**: argparse (in `cli.py`)
- **Cron parsing**: `cron_converter`
- **Git**: GitPython (`sub/git_manager.py`)
- **Package Manager**: uv

## Key Commands

```bash
# Development
uv run python -m notes_rag                      # Run MCP server + rebuild loop
uv run python -m notes_rag --verbose            # Enable DEBUG logging
uv run python -m notes_rag --rebuild-on-start   # Rebuild immediately on startup instead of waiting for first cron slot

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
export NOTES_BRANCH="main"
export CHROMA_PATH="./.chromadb"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export SERVER_HOST="0.0.0.0"
export SERVER_PORT=9099
export SERVER_WORKERS=1
export SERVER_TIMEOUT=30
export REBUILD_CRON="0 */6 * * *"
export TZ="Europe/Amsterdam"
export LOG_LEVEL="INFO"
```

Note: there is no `TOP_K` env var — the number of results per query (`num_docs`) is chosen by the MCP client on each `query_notes` call.

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
- **Storage receives list of `BaseExtractor` objects at initialization** for document ingestion
- **Extractors yield `TextNode` batches to Storage**: Each `BaseExtractor.get_nodes()` handles its own parsing and chunking, yielding batches of `TextNode` objects that `Storage.rebuild()` passes to `VectorStoreIndex.ainsert_nodes()` without re-parsing
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
  mcp_server.py            # FastMCP server exposing the `query_notes` tool
  logging_config.py        # Logging setup with StreamHandler
  extractor/
    abstract.py            # BaseExtractor (ABC)
    markdown_extractor.py  # MarkdownExtractor: git clone/pull, MarkdownNodeParser chunking
  sub/
    git_manager.py         # Git operations wrapper (clone, checkout, fetch, pull, status)
    storage_utils.py       # ChromaDB client factory + collection copy helper (used for atomic rebuild swap)
./tests/                   # Unit tests
  conftest.py              # pytest fixtures (setup_env, mock_config)
  test_storage.py
  extractor/
    test_markdown_extractor.py
  sub/
    test_git_manager.py
    test_storage_utils.py
pyproject.toml             # uv/hatch project config, CLI script `notes-rag`
uv.lock                    # Dependency lockfile
```

## Common Mistakes

- **Don't commit tokens**: `NOTES_TOKEN` never committed; use env vars
- **Match embedding models**: Embedding model must be identical between indexing and querying
- **Embedding**: `Storage` uses LlamaIndex `VectorStoreIndex.ainsert_nodes()` to generate embeddings — never manually
- **Embedding model**: `llama-index-embeddings-fastembed` (ONNX runtime) with `sentence-transformers/all-MiniLM-L6-v2`, passed to `VectorStoreIndex` via `embed_model` parameter
- **ChromaDB path**: Default is `.chromadb/`, overridable via `CHROMA_PATH`

## Daemon Workflow

**Startup**:

1. CLI parses args, loads `Config.from_env()`
2. `Storage` initialized with list of `BaseExtractor` objects (currently `[MarkdownExtractor(...)]`)
3. `MCPServer` initialized with `Storage`
4. `asyncio.gather()` starts: rebuild loop (cron-based) + `webserver.run()` (`FastMCP.run_streamable_http_async()`)

**Rebuild loop**:

1. Wait until next cron time (unless `--rebuild-on-start` was passed, which rebuilds immediately first)
2. For each extractor, iterate `get_nodes(batch_size=400)` and insert each batch into a temporary Chroma collection/index — the current main collection keeps serving queries throughout this phase
3. Acquire `asyncio.Lock`
4. Delete the old main collection, create a fresh one, and copy all records (metadata, documents, embeddings) from the temporary collection into it (`sub/storage_utils.copy_chroma_collection`)
5. Reset `_vector_store`/`_storage_context`/`_index` to reference the new main collection
6. Release lock
7. Errors: log and continue, old DB remains serving queries

**Query flow**:

1. MCP client calls the `query_notes` tool with `{query, num_docs}`
2. `Storage.search()` retrieves top-`num_docs` nodes
3. Format as delimited text:

   ```
   --- [source: file.md] ---
   chunk text
   --- [source: file2.md] ---
   another chunk
   ```

4. Return as `{"additional_context": "..."}`

**Shutdown** (SIGINT/SIGTERM/SIGABRT):

1. Signal handler cancels the main `asyncio` task
2. `asyncio.gather()` raises `CancelledError`, rebuild loop and MCP server both stop
3. Exit
