from pathlib import Path

import chromadb.config
from chromadb import Collection, Include, PersistentClient
from chromadb.api import ClientAPI


def create_chroma_client(path: Path) -> ClientAPI:
    """Create a ChromaDB persistent client (i.e., on local disk).

    Args:
        path: Path to the ChromaDB storage directory.

    Returns:
        ChromaDB PersistentClient instance.
    """
    client = PersistentClient(
        path=path,
        settings=chromadb.config.Settings(
            anonymized_telemetry=False,
        ),
    )
    return client


def copy_chroma_collection(source: Collection, dest: Collection, *, chunk_size: int = 100):
    """Copies all records from collection c1 to collection c2."""
    offset = 0
    limit = chunk_size
    included: Include = ["metadatas", "documents", "embeddings"]
    results = source.get(include=included, limit=limit, offset=offset)
    while results["ids"]:
        dest.add(
            ids=results["ids"],
            embeddings=results["embeddings"],
            documents=results["documents"],
            metadatas=results["metadatas"],
        )
        offset += len(results["ids"])

        results = source.get(include=included, limit=limit, offset=offset)

    return offset
