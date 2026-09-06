import pytest

from notes_rag.config import Config


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Set required environment variables for all tests."""
    monkeypatch.setenv("NOTES_DIR", "/tmp/notes")
    monkeypatch.setenv("NOTES_REPO_URL", "https://github.com/example/notes.git")
    monkeypatch.setenv("NOTES_REPO_BRANCH", "main")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")


@pytest.fixture
def mock_config(tmp_path):
    return Config(
        notes_repo_url="https://github.com/example/notes.git",
        notes_directory=tmp_path / "notes",
        notes_branch="main",
        chroma_path=tmp_path / "chromadb",
        embedding_model="BAAI/bge-small-en-v1.5",
        server_host="127.0.0.1",
        server_port=8000,
        server_workers=1,
        server_timeout=30,
        rebuild_cron="0 */6 * * *",
        timezone="Europe/Amsterdam",
    )
