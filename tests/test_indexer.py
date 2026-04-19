from pathlib import Path
from unittest.mock import MagicMock

import pytest

from notes_rag.indexer import Indexer
from notes_rag.storage import Storage
from notes_rag.sub.git_manager import GitManager


class TestIndexer:
    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock(spec=Storage)
        return storage

    def test_init(self, mock_storage, monkeypatch):
        """Verify behavior of init"""
        schedule = MagicMock(spec=str)
        notes_directory = MagicMock(spec=Path)
        notes_repo_url = MagicMock(spec=str)
        notes_repo_branch = MagicMock(spec=str)
        mock_git_manager = MagicMock(spec=GitManager)
        monkeypatch.setattr("notes_rag.indexer.GitManager", MagicMock(return_value=mock_git_manager))

        indexer = Indexer(mock_storage, schedule, notes_directory, notes_repo_url, notes_repo_branch)

        assert indexer._storage == mock_storage
        assert indexer._schedule == schedule
        assert indexer._notes_repo_url == notes_repo_url
        assert indexer._notes_repo_branch == notes_repo_branch
        assert indexer._git == mock_git_manager
