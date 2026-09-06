import logging
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import List
from uuid import UUID

import chromadb.config
from chromadb import PersistentClient
from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)

SQLITE_FILENAME = "chroma.sqlite3"


# NOTE: only update this to allow to use other backend for Chroma
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


def prune_orphan_segments(chroma_path: Path) -> List[Path]:
    """Remove HNSW segment directories no longer referenced by chroma.sqlite3

    Required to deal with older app version with incorrect rebuild mechanism.

    Args:
        chroma_path: Path to the ChromaDB storage directory

    Returns:
        List of directories that were removed
    """
    sqlite_path = Path(chroma_path).resolve() / SQLITE_FILENAME
    if not sqlite_path.is_file():
        logger.warning(f"No {SQLITE_FILENAME} under {str(chroma_path)!r}: skipping orphan segment prune")
        return []

    with closing(sqlite3.connect(f"{sqlite_path.as_uri()}?mode=ro", uri=True)) as connection:
        live_segments = {row[0] for row in connection.execute("SELECT id FROM segments")}

    removed: List[Path] = []
    for entry in sqlite_path.parent.iterdir():
        if entry.is_symlink() or not entry.is_dir() or entry.name in live_segments:
            continue
        try:
            # Segment directories are always named after the canonical form of their segment UUID
            if str(UUID(entry.name)) != entry.name:
                continue
        except ValueError:
            continue

        try:
            shutil.rmtree(entry)
        except OSError:
            logger.warning(f"Failed to remove orphan segment directory {str(entry)!r}", exc_info=True)
            continue
        removed.append(entry)

    return removed
