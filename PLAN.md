# RAG Implementation Plan

## Overview

This document outlines the plan for building a local vector retrieval service for markdown notes. The system indexes notes from a private GitHub repository into a local vector database and exposes a `/query_rag` HTTP endpoint for MCP servers to inject retrieved context into LLM prompts. This is retrieval-only — no LLM generation.

## Goals

- **Local-first**: All processing (indexing, embeddings) happens locally
- **No external providers**: Avoid cloud APIs
- **Efficient retrieval**: Chunk-based indexing with semantic search via ChromaDB
- **Scheduled rebuild**: Cron-based indexing of markdown notes from Git
- **HTTP API**: Expose `/query_rag` endpoint for MCP context injection
- **Retrieval-only**: Return formatted text, not LLM responses

## Architecture

### Components

```text
┌──────────────────────────────────────────────────────────────┐
│                   CLI (argparse)                             │
│   notes-rag → start daemon (rebuild + server)                │
│   notes-rag --only-rebuild → rebuild loop only               │
│   notes-rag --only-rebuild-once → single rebuild, exit       │
│   notes-rag --immediate → rebuild on startup                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────┴───────────┐
              │                        │
       ┌──────▼──────┐          ┌──────▼──────┐
       │ Rebuild     │          │ HTTPServer  │
       │ Loop        │          │ FastAPI     │
       │ (cron)      │          │ /query_rag  │
       └──────┬──────┘          └─────────────┘
              │
       ┌──────▼──────┐
       │ Storage     │
       │ ChromaDB    │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │ Extractors  │
       │ (abstract)  │
       └─────────────┘
```

### Component Descriptions

| Component             | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| **CLI**               | argparse-based entry point, daemon mode with rebuild loop + HTTP server |
| **Rebuild Loop**      | Cron-scheduled full rebuild, atomic swap for zero-downtime              |
| **HTTPServer**        | FastAPI server exposing `/query_rag` POST endpoint                      |
| **Storage**           | ChromaDB orchestration, extractor management, retrieval, atomic swap    |
| **Extractor**         | Abstract base class for document ingestion (passive, no threads)        |
| **MarkdownExtractor** | Concrete Extractor for `.md` files (clone/pull, parse, chunk)           |
| **ChromaDB**          | Local vector database for embeddings and metadata                       |

### Extractor Interface

The `BaseExtractor` class defines the interface for document ingestion. It is **passive** — it does not run as a thread and must be called by `Storage`. Each extractor handles its own setup (git clone, file discovery, parsing) and returns `ExtractorResult` with `TextNode` objects.

```python
@dataclass
class ExtractorResult:
    nodes: List[TextNode]
    files_processed: int

class BaseExtractor(ABC):
    @abstractmethod
    def get_nodes(self) -> ExtractorResult:
        """Process documents and return TextNodes for embedding."""
        ...
```

**Design Notes**:

- Passive: `Storage` calls `get_nodes()` on a schedule
- Full rebuild on each cycle (not incremental) for consistency
- Each extractor handles its own state (git repos, file discovery)
- No cleanup needed — state persists across restarts
- `Storage` receives extractors at initialization, no dynamic registration

## Daemon Lifecycle

### Startup

1. CLI parses args (`--verbose`, `--only-rebuild`, `--only-rebuild-once`, `--immediate`)
2. `Config.from_env()` loads all configuration
3. `Storage` initialized with list of `Extractor` objects
4. `HTTPServer` initialized with `Storage`
5. `asyncio.gather()` starts rebuild loop + `server.serve()`
6. If `--immediate` set, kick off initial rebuild

### Rebuild Loop

```text
1. Parse REBUILD_CRON env var (cron expression)
2. Calculate next run time
3. Wait until next run
4. Acquire asyncio.Lock
5. storage.rebuild() via asyncio.to_thread() (non-blocking)
6. Atomically swap _client, _chroma_collection, _vector_store, _storage_context, _index
7. Release lock
8. On error: log and continue (old DB remains serving)
```

**Atomic Swap**: During the lock window, queries see either the old or new state. The swap itself is microseconds — no long blocking.

### Query Flow

1. HTTP POST `/query_rag` with `{"query": "...", "num_docs": N}`
2. `Storage.search(query, top_k=num_docs)` retrieves top-k nodes
3. Format as delimited text:

   ```text
   --- [source: file.md] ---
   chunk text here
   --- [source: file2.md] ---
   another chunk
   ```

4. Return `{"response": "<delimited text>"}`

### Shutdown (SIGINT/SIGTERM)

1. Stop HTTP server (flush connections)
2. Stop rebuild loop
3. Exit

## HTTP Endpoint

### POST `/query_rag`

**Request**:

```json
{
  "query": "What is the architecture of the system?",
  "num_docs": 5
}
```

**Response**:

```json
{
  "response": "--- [source: notes/architecture.md] ---\nThe system uses FastAPI for the HTTP layer...\n\n--- [source: notes/deployment.md] ---\nDeployment is handled via uvicorn..."
}
```

**Response Format**: Delimited text with `[source: path]` markers per chunk. LLMs are trained on this pattern extensively — it is explicit, parsable, and avoids JSON encoding overhead.

## Configuration

### Config Dataclass

| Field             | Type | Required | Env Var           | Default            |
| ----------------- | ---- | -------- | ----------------- | ------------------ |
| `notes_repo_url`  | str  | Yes      | `NOTES_REPO_URL`  | —                  |
| `notes_directory` | Path | No       | `NOTES_DIRECTORY` | `./notes-cache`    |
| `chroma_path`     | Path | No       | `CHROMA_PATH`     | `./.chromadb`      |
| `top_k`           | int  | No       | `TOP_K`           | `5`                |
| `embedding_model` | str  | No       | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `server_host`     | str  | No       | `SERVER_HOST`     | `127.0.0.1`        |
| `server_port`     | int  | No       | `SERVER_PORT`     | `8000`             |
| `server_workers`  | int  | No       | `SERVER_WORKERS`  | `1`                |
| `server_timeout`  | int  | No       | `SERVER_TIMEOUT`  | `30`               |
| `rebuild_cron`    | str  | No       | `REBUILD_CRON`    | `"0 */6 * * *"`    |

### CLI Arguments

| Flag                  | Description                           |
| --------------------- | ------------------------------------- |
| `--verbose, -v`       | Enable DEBUG logging                  |
| `--only-rebuild`      | Run rebuild loop only, no HTTP server |
| `--only-rebuild-once` | Single rebuild, then exit             |
| `--immediate`         | Kick off rebuild on startup           |

## Cron Scheduling

Uses the `cron_converter` library to parse the `REBUILD_CRON` environment variable. Supported formats include standard crontab expressions (5 fields: minute, hour, day of month, month, day of week).

Default: `"0 */6 * * *"` (every 6 hours)

## Indexing Workflow

```text
1. CLI Trigger
   ↓
2. Storage.rebuild() called via asyncio.to_thread()
   ↓
3. Acquire asyncio.Lock
   ↓
4. For each Extractor in self._extractors:
   ↓
5. MarkdownExtractor.get_nodes():
   - Clone/pull notes repo via GitManager
   - Recursively find .md files in notes_directory
   - Parse with LlamaIndex MarkdownReader
   - Chunk with MarkdownNodeParser
   - Set source_file metadata on each TextNode
   ↓
6. Storage.add_nodes() calls VectorStoreIndex.insert_nodes()
   - Embeddings generated locally via HuggingFaceEmbedding
   ↓
7. Atomically swap _client, _chroma_collection, _vector_store, _storage_context, _index
   ↓
8. Release lock
   ↓
9. Log summary: N files processed, M nodes indexed
```

## Chunking Strategy

- **Parser**: `MarkdownNodeParser` (hierarchical — preserves document structure)
- **Metadata**: Each chunk includes `source_file` (relative path)
- **Format**: `MarkdownReader` preserves markdown formatting

**Rationale**: `MarkdownNodeParser` preserves document hierarchy (headings, lists, paragraphs) better than simple character-based splitting. This yields more coherent chunks for retrieval.

## Design Decisions

| Decision              | Choice              | Rationale                                             |
| --------------------- | ------------------- | ----------------------------------------------------- |
| **Framework**         | FastAPI             | Async-native, minimal boilerplate, auto-docs          |
| **Vector DB**         | ChromaDB            | Local-first, Python-native, minimal setup             |
| **Embedding Model**   | `all-MiniLM-L6-v2`  | Fast, lightweight (~90MB), good quality for retrieval |
| **Extractor Pattern** | Passive, list-based | Simple, no threading, each extractor owns its state   |
| **Rebuild**           | Full + atomic swap  | Consistent indexing, zero-downtime queries            |
| **Cron Library**      | `cron_converter`    | Lightweight, standard crontab support                 |
| **Git Auth**          | Token in URL        | Simpler than credential helper setup                  |
| **Response Format**   | Delimited text      | Standard for MCP tool results, LLM-friendly           |

## Future Improvements

1. **More extractors**: Support PDF, DOCX, etc. via new `BaseExtractor` subclasses
2. **Incremental indexing**: Only index new/modified files, not full rebuild
3. **Hybrid search**: Combine keyword (BM25) and semantic search
4. **Query rewriting**: Improve query embeddings for better retrieval
5. **Citation ranking**: Prioritize chunks from more recent/authoritative sources
6. **Multi-model support**: Swap embedding models without code changes

## Security Considerations

- **NOTES_TOKEN** should never be committed; use `.env` or shell export
- **notes-directory/** should be added to `.gitignore` to prevent accidental commits
- Token format: `https://ghp_xx@github.com/user/repo.git` (GitHub Personal Access Token)
- Consider rotating tokens periodically

## Testing Strategy

- **Unit tests**: Isolate config loading, storage operations, CLI argument parsing
- **Integration tests**: Test with sample markdown files and mock ChromaDB
- **Mock convention**: Use `unittest.mock.MagicMock` for complex objects
- **Patching conventions**: Use Pytest's `monkeypatch` built-in fixture
- **Fixtures**: `conftest.py` provides `setup_env` (autouse) and `mock_config`

## Performance Considerations

- **Embedding model**: `all-MiniLM-L6-v2` is fast (~1ms/chunk) and memory-efficient
- **ChromaDB**: Persists to disk at `CHROMA_PATH`
- **Atomic swap**: Lock held only for microseconds during reference swap
- **asyncio.to_thread()**: Blocking git/embedding operations run off the event loop
- **Rebuild frequency**: Configurable via cron (default every 6 hours)

## Implementation Order

1. **config.py** — Centralized Config dataclass, all env vars
2. **storage.py** — ChromaDB orchestration, extractor management, atomic swap rebuild, retrieval
3. **extractor/abstract.py** — `BaseExtractor`, `ExtractorResult`
4. **extractor/markdown_extractor.py** — `MarkdownExtractor` (git, parse, chunk)
5. **webserver.py** — `HTTPServer`, `/query_rag` endpoint, delimited text formatting
6. **cli.py** — Argparse, daemon loop (`asyncio.gather`), cron scheduling, flags
7. **sub/git_manager.py** — Git operations wrapper (clone, checkout, fetch, pull, status)
8. **pyproject.toml** — Dependencies (add `cron_converter`)
9. **tests/** — Unit tests for storage, webserver, CLI, git_manager
