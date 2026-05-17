from pathlib import Path

import pytest

from notes_rag.config import Config


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Set required environment variables for all tests."""
    monkeypatch.setenv("NOTES_DIR", "/tmp/notes")
    monkeypatch.setenv("NOTES_REPO_URL", "https://github.com/example/notes.git")
    monkeypatch.setenv("NOTES_REPO_BRANCH", "main")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")


@pytest.fixture
def mock_config():
    return Config(
        ollama_host="http://localhost:11434",
        notes_repo_url="https://github.com/example/notes.git",
        notes_cache_dir=Path("/tmp/notes"),
        chunk_size=1000,
        top_k=5,
        embedding_model="BAAI/bge-small-en-v1.5",
        log_level="WARNING",
    )
