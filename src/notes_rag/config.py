import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def get_required_env(variable_name: str) -> str:
    got = os.getenv(variable_name)
    if not got:
        raise RuntimeError(f"Environment variable '{variable_name}' is not set!")
    return got


DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "notes_repo_url": None,
    "notes_cache_dir": "./notes-cache",
    "chunk_size": 1000,
    "top_k": 5,
    "embedding_model": "all-MiniLM-L6-v2",
    "log_level": "INFO",
}


@dataclass
class Config:
    """Configuration class for the Notes RAG application.

    Reads configuration from environment variables with sensible defaults.
    Required environment variables:
        - OLLAMA_HOST: Base URL for Ollama API
        - NOTES_REPO_URL: GitHub repository URL with embedded token

    Optional environment variables with defaults:
        - NOTES_CACHE_DIR: Local cache directory (default: "./notes-cache")
        - CHUNK_SIZE: Maximum chunk size in characters (default: 1000)
        - TOP_K: Number of chunks to retrieve per query (default: 5)
        - EMBEDDING_MODEL: Sentence transformers model (default: "all-MiniLM-L6-v2")
        - LOG_LEVEL: Logging level (default: "INFO")
    """

    ollama_host: str
    notes_repo_url: str
    notes_cache_dir: Path
    chunk_size: int
    top_k: int
    embedding_model: str
    log_level: str

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        load_dotenv(env_file)

        notes_repo_url = get_required_env("NOTES_REPO_URL")

        ollama_host = os.getenv("OLLAMA_HOST", "not set")
        notes_cache_dir = Path(os.getenv("NOTES_CACHE_DIR", DEFAULT_CONFIG["notes_cache_dir"])).resolve()
        chunk_size = int(os.getenv("CHUNK_SIZE", DEFAULT_CONFIG["chunk_size"]))
        top_k = int(os.getenv("TOP_K", str(DEFAULT_CONFIG["top_k"])))
        embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_CONFIG["embedding_model"])
        log_level = os.getenv("LOG_LEVEL", DEFAULT_CONFIG["log_level"])

        return cls(
            ollama_host=ollama_host,
            notes_repo_url=notes_repo_url,
            notes_cache_dir=notes_cache_dir,
            chunk_size=chunk_size,
            top_k=top_k,
            embedding_model=embedding_model,
            log_level=log_level,
        )
