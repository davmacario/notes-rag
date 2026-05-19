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
    "notes_directory": "./notes-cache",
    "notes_branch": "main",
    "chroma_path": "./.chromadb",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "server_host": "0.0.0.0",
    "server_port": 9099,
    "server_workers": 1,
    "server_timeout": 30,
    "rebuild_cron": "0 */6 * * *",
    "timezone": "Europe/Amsterdam",
}


@dataclass
class Config:
    """Configuration class for the Notes RAG application.

    Reads configuration from environment variables with sensible defaults.
    Required environment variables:
        - NOTES_REPO_URL: GitHub repository URL with embedded token

    Optional environment variables with defaults:
        - NOTES_DIRECTORY: Local cache directory (default: "./notes-cache")
        - NOTES_BRANCH: Git branch of the notes repository (default: "main")
        - CHROMA_PATH: Path to ChromaDB storage (default: "./.chromadb")
        - EMBEDDING_MODEL: Embedding model name (default: "sentence-transformers/all-MiniLM-L6-v2")
        - SERVER_HOST: HTTP server host (default: "0.0.0.0")
        - SERVER_PORT: HTTP server port (default: 9099)
        - SERVER_WORKERS: HTTP server workers (default: 1)
        - SERVER_TIMEOUT: HTTP server timeout (default: 30)
        - REBUILD_CRON: Cron expression for rebuild schedule (default: "0 */6 * * *")
        - TZ: timezone
    """

    notes_repo_url: str
    notes_directory: Path
    notes_branch: str
    chroma_path: Path
    embedding_model: str
    server_host: str
    server_port: int
    server_workers: int
    server_timeout: int
    rebuild_cron: str
    timezone: str

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        load_dotenv(env_file)

        notes_repo_url = get_required_env("NOTES_REPO_URL")
        notes_directory = Path(os.getenv("NOTES_DIRECTORY", DEFAULT_CONFIG["notes_directory"])).resolve()
        notes_branch = os.getenv("NOTES_BRANCH", DEFAULT_CONFIG["notes_branch"])
        chroma_path = Path(os.getenv("CHROMA_PATH", DEFAULT_CONFIG["chroma_path"])).resolve()
        embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_CONFIG["embedding_model"])
        server_host = os.getenv("SERVER_HOST", DEFAULT_CONFIG["server_host"])
        server_port = int(os.getenv("SERVER_PORT", str(DEFAULT_CONFIG["server_port"])))
        server_workers = int(os.getenv("SERVER_WORKERS", str(DEFAULT_CONFIG["server_workers"])))
        server_timeout = int(os.getenv("SERVER_TIMEOUT", str(DEFAULT_CONFIG["server_timeout"])))
        rebuild_cron = os.getenv("REBUILD_CRON", DEFAULT_CONFIG["rebuild_cron"])
        timezone = os.getenv("TZ", DEFAULT_CONFIG["timezone"])

        return cls(
            notes_repo_url=notes_repo_url,
            notes_directory=notes_directory,
            notes_branch=notes_branch,
            chroma_path=chroma_path,
            embedding_model=embedding_model,
            server_host=server_host,
            server_port=server_port,
            server_workers=server_workers,
            server_timeout=server_timeout,
            rebuild_cron=rebuild_cron,
            timezone=timezone,
        )
