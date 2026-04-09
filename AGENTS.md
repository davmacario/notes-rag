# Agents Guide

## Project Type
Python RAG (Retrieval Augmented Generation) system for local markdown note retrieval with Ollama-powered LLM responses.

## Key Facts

- **Stack**: Python + LlamaIndex + ChromaDB + sentence-transformers + Ollama
- **Indexing**: `./rag index` — clones private notes repo, chunks markdown, generates embeddings
- **Querying**: `./rag query "question"` — searches embeddings, returns cited answer from local LLM
- **Embedding model**: `all-MiniLM-L6-v2` (768-dim, fast, local)
- **Vector DB**: ChromaDB (local, persisted in `.chromadb/`)
- **Notes cache**: `.gitignore`d at `./notes-cache/` (cloned from GitHub via token auth)

## Environment Variables (required)

```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="mistral"
export NOTES_REPO_URL="https://ghp_xxxxx@github.com/user/notes.git"
```

## Development Commands

After `pip install -r requirements.txt`:

- **Re-index notes**: `python main.py index`
- **Query**: `python main.py query "your question"`
- **Clear cache**: `python main.py query --clear-cache`

## Important Conventions

- **Prompt includes citation instructions**: LLM must cite sources as `[source: path/to/file.md]`
- **Simple chunking**: 1000-char chunks with 200-overlap (no header hierarchy yet)
- **GitHub auth**: Token embedded in `NOTES_REPO_URL` (not separate variable)
- **No external providers**: All embedding and LLM inference is local

## File Structure

```
main.py          # CLI entry point (click)
config.py        # Environment loading
indexer.py       # Git clone, parse, chunk, embed
query.py         # Search, prompt, Ollama generation
storage.py       # ChromaDB wrapper
requirements.txt # Dependencies
```

## Common Mistakes to Avoid

- Don't commit `NOTES_TOKEN` or `./notes-cache/`
- Don't assume `notes-cache/` exists — indexer clones it fresh
- Don't skip Ollama setup — LLM responses require running `ollama serve`
- Don't change embedding model without also updating query module (must match)

## Next Steps (from PLAN.md)

See `PLAN.md` for detailed workflow and future improvements.
