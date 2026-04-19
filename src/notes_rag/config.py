import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default configuration values
DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "ollama_model": "gpt-oss:20b",
    "notes_repo_url": None,
    "notes_cache_dir": "./notes-cache",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "embedding_model": "all-MiniLM-L6-v2",
}


@dataclass
class Config:
    """Configuration class for the Notes RAG application.

    Reads configuration from environment variables with sensible defaults.
    Required environment variables:
        - OLLAMA_HOST: Base URL for Ollama API
        - NOTES_REPO_URL: GitHub repository URL with embedded token

    Optional environment variables with defaults:
        - OLLAMA_MODEL: LLM model name (default: "mistral")
        - NOTES_CACHE_DIR: Local cache directory (default: "./notes-cache")
        - CHUNK_SIZE: Maximum chunk size in characters (default: 1000)
        - CHUNK_OVERLAP: Overlap between chunks (default: 200)
        - TOP_K: Number of chunks to retrieve per query (default: 5)
        - EMBEDDING_MODEL: Sentence transformers model (default: "all-MiniLM-L6-v2")
    """

    ollama_host: str
    ollama_model: str
    notes_repo_url: str | None
    notes_cache_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        # Required environment variables
        self.ollama_host = self._get_required_env("OLLAMA_HOST")
        self.ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_CONFIG["ollama_model"])
        self.notes_repo_url = os.getenv("NOTES_REPO_URL")

        # Optional environment variables with defaults
        self.notes_cache_dir = Path(
            os.getenv("NOTES_CACHE_DIR", DEFAULT_CONFIG["notes_cache_dir"])
        )
        self.chunk_size = int(os.getenv("CHUNK_SIZE", DEFAULT_CONFIG["chunk_size"]))
        self.chunk_overlap = int(
            os.getenv("CHUNK_OVERLAP", DEFAULT_CONFIG["chunk_overlap"])
        )
        self.top_k = int(os.getenv("TOP_K", str(DEFAULT_CONFIG["top_k"])))
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", DEFAULT_CONFIG["embedding_model"]
        )

        # Validate required variables
        self._validate()

    def _get_required_env(self, name: str) -> str:
        """Get a required environment variable or raise ValueError."""
        value = os.getenv(name)
        if not value:
            raise ValueError(
                f"Required environment variable '{name}' is not set. "
                f"Please set it in your environment or .env file."
            )
        return value

    def _validate(self) -> None:
        """Validate configuration values."""
        if not self.ollama_host:
            raise ValueError(
                "OLLAMA_HOST is required but not set. "
                "Please set it to your Ollama API base URL (e.g., http://localhost:11434)"
            )
        if not self.notes_repo_url:
            raise ValueError(
                "NOTES_REPO_URL is required but not set. "
                "Please set it to your GitHub notes repository URL with embedded token."
            )

    @property
    def notes_cache_path(self) -> Path:
        """Get the absolute path to the notes cache directory."""
        return Path(self.notes_cache_dir).resolve()
