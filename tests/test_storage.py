import os
import shutil
import sys

import pytest
from llama_index.core import Document

from notes_rag.storage import Storage, create_chroma_client


class TestCreateChromaClient:
    """Test create_chroma_client function."""

    def test_creates_chroma_client(self, tmp_path):
        """Test that ChromaDB client is created with correct path."""
        chroma_path = tmp_path / "chroma_db"
        client = create_chroma_client(str(chroma_path))

        # Client should be created
        assert client is not None
        # Collection doesn't exist yet, that's expected
        # Just verify client was created successfully


class TestStorage:
    """Test Storage class."""

    @pytest.fixture(autouse=True)
    def setup_storage(self, tmp_path):
        """Set up test fixtures using tmp_path."""
        self.storage = Storage(chroma_path=str(tmp_path / "chroma"))
        self.storage.clear()
        yield
        # Cleanup after test
        test_path = tmp_path / "chroma"
        if test_path.exists():
            shutil.rmtree(test_path)

    def test_add_documents(self):
        """Test adding documents to ChromaDB."""
        docs = [
            Document(text="This is test document 1", metadata={"source_file": "test1.md"}),
            Document(text="This is test document 2", metadata={"source_file": "test2.md"}),
        ]

        self.storage.add_documents(docs)
        collection = self.storage._collection
        assert collection.count() == 2

    def test_add_documents_with_metadata(self):
        """Test that metadata is preserved when adding documents."""
        doc = Document(text="Test content", metadata={"source_file": "example.md", "chunk_index": 0})

        self.storage.add_documents([doc])

        # Retrieve and verify metadata
        collection = self.storage._collection
        result = collection.get(include=["metadatas"])
        assert result["metadatas"][0] == {"source_file": "example.md", "chunk_index": 0}

    def test_search(self):
        """Test searching for similar documents."""
        docs = [
            Document(text="Machine learning is a subset of AI", metadata={"source_file": "ai.md"}),
            Document(text="Deep learning uses neural networks", metadata={"source_file": "dl.md"}),
            Document(text="Python is a programming language", metadata={"source_file": "python.md"}),
        ]

        self.storage.add_documents(docs)

        # Search for "machine learning" related content
        results = self.storage.search("machine learning", top_k=2)
        assert len(results) == 2
        assert isinstance(results, list)
        assert all(hasattr(r, "text") for r in results)

    def test_search_empty_query(self):
        """Test that search handles empty queries gracefully."""
        docs = [Document(text="Test content", metadata={"source_file": "test.md"})]
        self.storage.add_documents(docs)

        # Empty query should return empty results
        results = self.storage.search("", top_k=5)
        assert len(results) == 0

    def test_clear(self):
        """Test clearing all documents from the collection."""
        docs = [
            Document(text="Test 1", metadata={"source_file": "test1.md"}),
            Document(text="Test 2", metadata={"source_file": "test2.md"}),
        ]

        self.storage.add_documents(docs)
        assert self.storage._collection.count() == 2

        self.storage.clear()
        assert self.storage._collection.count() == 0

    def test_search_limit(self):
        """Test that search respects top_k parameter."""
        docs = [Document(text=f"Test content {i}", metadata={"source_file": f"test{i}.md"}) for i in range(10)]

        self.storage.add_documents(docs)

        # Request only 3 results
        results = self.storage.search("test", top_k=3)
        assert len(results) <= 3
