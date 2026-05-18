from unittest.mock import MagicMock, call

import pytest
from chromadb import Collection
from chromadb.api import ClientAPI

from notes_rag.storage import create_chroma_client
from notes_rag.sub.storage_utils import copy_chroma_collection


class TestCreateChromaClient:
    """Test create_chroma_client function."""

    def test_creates_chroma_client(self, tmp_path):
        """Test that ChromaDB client is created with correct path."""
        chroma_path = tmp_path / "chroma_db"
        client = create_chroma_client(chroma_path)

        # Client should be created
        assert client is not None
        assert isinstance(client, ClientAPI)


class TestCopyChromaCollection:
    """Test copy_chroma_collection function."""

    @pytest.fixture
    def mock_collections(self):
        return {
            "src": MagicMock(spec=Collection),
            "dst": MagicMock(spec=Collection),
        }

    def test_copy_chroma_collection(self, mock_collections):
        empty_get_result = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
        get_result = {
            "ids": ["aaa", "bbb", "ccc"],
            "documents": ["Hello, world!", "How are you", "My name is Davide"],
            "embeddings": [[-1.903, 2.3109], [-3.023, 0.41], [2.903, 8.30]],
            "metadatas": [{"source": "tmp.txt"}, {"source": "dir/file2.txt"}, {"source": "dir/file3.txt"}],
        }
        mock_collections["src"].get.side_effect = [get_result, empty_get_result]
        included = ["metadatas", "documents", "embeddings"]

        count = copy_chroma_collection(mock_collections["src"], mock_collections["dst"], chunk_size=4)

        assert count == 3
        mock_collections["dst"].add.assert_called_once_with(
            ids=get_result["ids"],
            embeddings=get_result["embeddings"],
            documents=get_result["documents"],
            metadatas=get_result["metadatas"],
        )
        mock_collections["src"].get.assert_has_calls(
            [
                call(include=included, limit=4, offset=0),
                call(include=included, limit=4, offset=3)
            ]
        )
