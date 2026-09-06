import logging
import shutil
import sqlite3
import uuid
from pathlib import Path

import pytest
from chromadb.api import ClientAPI

from notes_rag.sub.storage_utils import create_chroma_client, prune_orphan_segments


class TestCreateChromaClient:
    """Test create_chroma_client function."""

    def test_creates_chroma_client(self, tmp_path):
        """Test that ChromaDB client is created with correct path."""
        chroma_path = tmp_path / "chroma_db"
        client = create_chroma_client(chroma_path)

        # Client should be created
        assert client is not None
        assert isinstance(client, ClientAPI)


class TestPruneOrphanSegments:
    """Test prune_orphan_segments function"""

    @pytest.fixture
    def chroma_dir(self, tmp_path) -> Path:
        """ChromaDB-like directory holding a real (but minimal) `segments` table."""
        path = tmp_path / "chroma_db"
        path.mkdir()
        with sqlite3.connect(path / "chroma.sqlite3") as connection:
            connection.execute("create table segments (id text primary key, type text, scope text, collection text)")
        return path

    @staticmethod
    def _add_live_segment(chroma_dir: Path) -> Path:
        """Register a segment in the DB and create its directory."""
        segment_id = str(uuid.uuid4())
        with sqlite3.connect(chroma_dir / "chroma.sqlite3") as connection:
            connection.execute(
                "insert into segments values (?, 'hnsw', 'VECTOR', ?)", (segment_id, str(uuid.uuid4()))
            )
        segment_dir = chroma_dir / segment_id
        segment_dir.mkdir()
        (segment_dir / "data_level0.bin").write_bytes(b"x" * 32)
        return segment_dir

    @staticmethod
    def _add_orphan(chroma_dir: Path) -> Path:
        """Create a segment directory that no DB row refers to."""
        orphan_dir = chroma_dir / str(uuid.uuid4())
        orphan_dir.mkdir()
        (orphan_dir / "data_level0.bin").write_bytes(b"x" * 32)
        return orphan_dir

    def test_removes_orphan_directories(self, chroma_dir):
        live = self._add_live_segment(chroma_dir)
        orphan = self._add_orphan(chroma_dir)

        removed = prune_orphan_segments(chroma_dir)

        assert removed == [orphan]
        assert not orphan.exists()
        assert live.is_dir()

    def test_keeps_live_directories(self, chroma_dir):
        live_dirs = [self._add_live_segment(chroma_dir) for _ in range(3)]

        removed = prune_orphan_segments(chroma_dir)

        assert removed == []
        assert all(d.is_dir() for d in live_dirs)

    def test_ignores_non_uuid_directories(self, chroma_dir):
        for name in ("hnsw", "notes-cache", "12345", "{" + str(uuid.uuid4()) + "}"):
            (chroma_dir / name).mkdir()

        removed = prune_orphan_segments(chroma_dir)

        assert removed == []
        assert len(list(chroma_dir.iterdir())) == 5  # 4 directories + chroma.sqlite3

    def test_ignores_files(self, chroma_dir):
        stray_file = chroma_dir / str(uuid.uuid4())
        stray_file.write_text("not a segment")

        removed = prune_orphan_segments(chroma_dir)

        assert removed == []
        assert stray_file.is_file()
        assert (chroma_dir / "chroma.sqlite3").is_file()

    def test_missing_database_deletes_nothing(self, chroma_dir):
        orphan = self._add_orphan(chroma_dir)
        (chroma_dir / "chroma.sqlite3").unlink()

        removed = prune_orphan_segments(chroma_dir)

        assert removed == []
        assert orphan.is_dir()

    def test_missing_directory_returns_empty(self, tmp_path):
        assert prune_orphan_segments(tmp_path / "does-not-exist") == []

    def test_does_not_write_to_the_database(self, chroma_dir):
        self._add_orphan(chroma_dir)
        database = chroma_dir / "chroma.sqlite3"
        before = database.stat()

        prune_orphan_segments(chroma_dir)

        assert database.stat().st_size == before.st_size
        assert database.stat().st_mtime_ns == before.st_mtime_ns

    def test_failure_does_not_abort_sweep(self, monkeypatch, caplog, chroma_dir):
        failing = self._add_orphan(chroma_dir)
        other = self._add_orphan(chroma_dir)
        real_rmtree = shutil.rmtree

        def flaky_rmtree(path, *args, **kwargs):
            if Path(path) == failing:
                raise PermissionError("locked")
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr("notes_rag.sub.storage_utils.shutil.rmtree", flaky_rmtree)

        with caplog.at_level(logging.WARNING):
            removed = prune_orphan_segments(chroma_dir)

        assert removed == [other]
        assert failing.is_dir()
        assert not other.exists()
        assert str(failing) in caplog.text
