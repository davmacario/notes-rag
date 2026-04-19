# Agents Guide

## Project Type

Python RAG (Retrieval Augmented Generation) system for local markdown note retrieval with Ollama-powered LLM responses.

## Stack

- **Framework**: LlamaIndex (document ingestion, RAG orchestration)
- **Vector DB**: ChromaDB (local, persisted in `.chromadb/`)
- **Embeddings**: `sentence-transformers` with `all-MiniLM-L6-v2` (768-dim)
- **LLM**: Ollama API (`${OLLAMA_HOST}/api/generate`)
- **CLI**: Click
- **Package Manager**: uv

## Key Commands

```bash
# Development
uv run python main.py index              # Re-index notes from GitHub repo
uv run python main.py query "question"   # Search & generate answer with citations
uv run python main.py query "question" --clear-cache  # Reset vector DB

# Setup
uv sync                                  # Install dependencies
```

## Environment Variables

Required:

```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="mistral"
export NOTES_REPO_URL="https://ghp_xxxxx@github.com/user/notes.git"
```

Optional (defaults shown):

```bash
export NOTES_CACHE_DIR="./notes-cache"        # Cloned repo location
export CHUNK_SIZE=1000                         # Chunk size in chars
export CHUNK_OVERLAP=200                       # Overlap between chunks
export TOP_K=5                                 # Chunks retrieved per query
```

## Development Workflow

- **Test-Driven Development**: Write tests first for all new features, then implement
- **Testing**: Use pytest for unit/integration tests, each source file has an associated test file
- **Linting**: `uv run ruff check`
- **Type checking**: `uv run pyright`
- **Run tests**: `uv run pytest tests/`

## Important Conventions

- **Citations**: LLM must use format `[source: path/to/file.md]` inline
- **Chunking**: Simple 1000-char chunks with 200-overlap (no header hierarchy)
- **GitHub auth**: Token embedded in URL as `https://ghp_xxxx@github.com/...`
- **Local only**: All embedding generation and LLM inference happens locally
- **ChromaDB persistence**: Stored in `.chromadb/` directory (gitignored)
- **Coding conventions**:
  - Avoid inline comments when the logic is self-explainatory
  - Avoid docstrings at the top of python files. The only docstrings present should be the ones for functions, classes, methods, and dataclass attributes
  - Docstrings should be concise
- **Testing conventions**:
  - Use `pytest` as much as possible, especially built-in fixtures (e.g., `monkeypatch`, `caplog`)
  - For mocking objects, use `unittest.mock.MagicMock`

## File Structure

```text
./src/notes_rag/     # Application source code
  cli.py             # CLI entry point with `index` and `query` commands
  config.py          # Environment loading and configuration
  indexer.py         # Git clone/pull, markdown parsing, chunking, embedding
  query.py           # Search ChromaDB, assemble prompt, call Ollama API
  storage.py         # ChromaDB initialization and CRUD operations
./tests/             # Unit tests (each source file has a test file)
  test_config.py
  test_storage.py
  test_indexer.py
  test_query.py
pyproject.toml       # uv project configuration
uv.lock              # Dependency lockfile
PLAN.md              # Detailed workflow and design decisions
main.py              # Main entrypoint of the application
```

## Common Mistakes

- **Don't commit tokens**: `NOTES_TOKEN` never committed; use env vars
- **Don't ignore notes-cache**: `.gitignore`d at `./notes-cache/`
- **Match embedding models**: Embedding model must be identical between indexing and querying
- **Clone first**: Indexer clones notes repo fresh each run (doesn't assume existing cache)

## RAG Workflow

**Indexing** (`./rag index`):

1. Clone/pull GitHub repo to `./notes-cache/`
2. Parse `.md` files with LlamaIndex MarkdownReader
3. Chunk by 1000 chars (200 overlap)
4. Generate embeddings via sentence-transformers
5. Store in ChromaDB with source file metadata

**Querying** (`./rag query "question"`):

1. Embed query with same model
2. Search ChromaDB for top-k similar chunks
3. Assemble prompt with context + question
4. Call Ollama API for response with citation instructions
5. Print answer with inline citations

## Next Steps

See `PLAN.md` for detailed workflow, prompt templates, and future improvements.
