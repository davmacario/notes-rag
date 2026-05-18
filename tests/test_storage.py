import logging
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from chromadb import Collection
from chromadb.api import ClientAPI
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from notes_rag.config import Config
from notes_rag.extractor.abstract import ExtractorResult
from notes_rag.extractor.markdown_extractor import MarkdownExtractor
from notes_rag.storage import Storage, create_chroma_client


class TestCreateChromaClient:
    """Test create_chroma_client function."""

    def test_creates_chroma_client(self, tmp_path):
        """Test that ChromaDB client is created with correct path."""
        chroma_path = tmp_path / "chroma_db"
        client = create_chroma_client(chroma_path)

        # Client should be created
        assert client is not None
        assert isinstance(client, ClientAPI)


class TestStorage:
    """Test Storage class."""

    @pytest.fixture
    def mocked_chroma_collection(self):
        return MagicMock(spec=Collection)

    @pytest.fixture
    def mocked_chroma_client(self, mocked_chroma_collection):
        client = MagicMock(spec=ClientAPI)
        client.get_or_create_collection.return_value = mocked_chroma_collection
        client.create_collection.return_value = mocked_chroma_collection
        return client

    @pytest.fixture
    def mocked_md_extractor(self):
        extractor = MagicMock(spec=MarkdownExtractor)
        # TODO
        return extractor

    @pytest.fixture
    def mocked_chroma_vector_store(self):
        chroma_vector_store = MagicMock(spec=ChromaVectorStore)
        return chroma_vector_store

    @pytest.fixture
    def mocked_storage_context(self):
        storage_context = MagicMock(spec=StorageContext)

        return storage_context

    @pytest.fixture
    def mocked_vector_store_index(self):
        vector_store_index = MagicMock(spec=VectorStoreIndex)
        vector_store_index.ainsert_nodes = AsyncMock()
        vector_store_index.ainsert_nodes.return_value = None
        return vector_store_index

    @pytest.fixture
    def mocked_huggingface_embedding(self):
        return MagicMock(spec=HuggingFaceEmbedding)

    @pytest.fixture
    def storage(
        self,
        monkeypatch,
        mock_config: Config,
        mocked_chroma_client: ClientAPI,
        mocked_md_extractor: MarkdownExtractor,
        mocked_chroma_vector_store: ChromaVectorStore,
        mocked_storage_context: StorageContext,
        mocked_vector_store_index: VectorStoreIndex,
        mocked_huggingface_embedding: HuggingFaceEmbedding,
    ) -> Storage:
        """Instance of Storage with mocked components"""
        monkeypatch.setattr("notes_rag.storage.create_chroma_client", lambda *_: mocked_chroma_client)
        monkeypatch.setattr("notes_rag.storage.ChromaVectorStore", lambda **_: mocked_chroma_vector_store)
        monkeypatch.setattr("notes_rag.storage.StorageContext.from_defaults", lambda **_: mocked_storage_context)
        monkeypatch.setattr("notes_rag.storage.VectorStoreIndex", lambda **_: mocked_vector_store_index)
        monkeypatch.setattr("notes_rag.storage.HuggingFaceEmbedding", lambda **_: mocked_huggingface_embedding)
        return Storage(mock_config, [mocked_md_extractor])

    # --- __init__ ---

    def test_init(
        self,
        monkeypatch,
        caplog,
        mock_config: Config,
        mocked_chroma_client: ClientAPI,
        mocked_md_extractor: MarkdownExtractor,
        mocked_chroma_vector_store: ChromaVectorStore,
        mocked_storage_context: StorageContext,
        mocked_vector_store_index: VectorStoreIndex,
        mocked_huggingface_embedding: HuggingFaceEmbedding,
        mocked_chroma_collection: Collection,
    ):
        mock_create_chroma_client = MagicMock(return_value=mocked_chroma_client)
        monkeypatch.setattr("notes_rag.storage.create_chroma_client", mock_create_chroma_client)
        mock_chroma_vector_store_factory = MagicMock(return_value=mocked_chroma_vector_store)
        monkeypatch.setattr("notes_rag.storage.ChromaVectorStore", mock_chroma_vector_store_factory)
        mock_storage_context_from_defaults = MagicMock(return_value=mocked_storage_context)
        monkeypatch.setattr("notes_rag.storage.StorageContext.from_defaults", mock_storage_context_from_defaults)
        mock_vector_store_index_factory = MagicMock(return_value=mocked_vector_store_index)
        monkeypatch.setattr("notes_rag.storage.VectorStoreIndex", mock_vector_store_index_factory)
        mock_huggingface_embedding_factory = MagicMock(return_value=mocked_huggingface_embedding)
        monkeypatch.setattr("notes_rag.storage.HuggingFaceEmbedding", mock_huggingface_embedding_factory)

        with caplog.at_level(logging.INFO):
            storage = Storage(mock_config, [mocked_md_extractor])

        assert storage._config == mock_config
        assert storage._extractors == [mocked_md_extractor]
        mock_create_chroma_client.assert_called_once_with(mock_config.chroma_path)
        assert storage._client == mocked_chroma_client
        storage._client.get_or_create_collection.assert_called_once_with(name=storage.COLLECTION_NAME)
        assert storage._chroma_collection == mocked_chroma_collection
        mock_chroma_vector_store_factory.assert_called_once_with(
            chroma_collection=mocked_chroma_collection, collection_name=storage.COLLECTION_NAME
        )
        assert storage._vector_store == mocked_chroma_vector_store
        mock_storage_context_from_defaults.assert_called_once_with(vector_store=mocked_chroma_vector_store)
        assert storage._storage_context == mocked_storage_context
        mock_huggingface_embedding_factory.assert_called_once_with(model_name=mock_config.embedding_model)
        assert storage._embed_model == mocked_huggingface_embedding
        mock_vector_store_index_factory.assert_called_once_with(
            use_async=True, storage_context=mocked_storage_context, embed_model=mocked_huggingface_embedding
        )
        assert f"Initialized ChromaDB at {str(mock_config.chroma_path)!r}" in caplog.text

    # --- add_nodes ---

    async def test_add_nodes(self, caplog, storage):
        """Test adding TextNodes to storage generates embeddings and stores in ChromaDB."""
        nodes = [
            TextNode(text="Node 1 content about machine learning", metadata={"source_file": "test1.md"}),
            TextNode(text="Node 2 content about neural networks", metadata={"source_file": "test2.md"}),
        ]

        with caplog.at_level(logging.INFO):
            await storage.add_nodes(nodes)

        storage._index.ainsert_nodes.assert_called_once_with(nodes)

    # --- search ---

    async def test_search(self, caplog, storage):
        """Test searching for similar documents."""
        mock_retriever = AsyncMock(spec=BaseRetriever)
        nodes = [
            TextNode(text="Node 1 content about machine learning", metadata={"source_file": "test1.md"}),
            TextNode(text="Node 2 content about neural networks", metadata={"source_file": "test2.md"}),
        ]
        mock_retriever.aretrieve.return_value = [
            NodeWithScore(node=nodes[0], score=0.92),
            NodeWithScore(node=nodes[1], score=0.85),
        ]
        storage._index.as_retriever.return_value = mock_retriever

        # Search for "machine learning" related content
        with caplog.at_level(logging.DEBUG):
            results = await storage.search("machine learning\n", top_k=2)

        assert results == nodes
        assert "Queried 2 nodes for query: 'machine learning'" in caplog.text
        storage._index.as_retriever.assert_called_once_with(similarity_top_k=2)
        mock_retriever.aretrieve.assert_called_once_with("machine learning\n")

    async def test_search_empty_query(self, storage):
        """Test that search handles empty queries gracefully."""
        results = await storage.search("\n\r", top_k=5)
        assert len(results) == 0

    # --- clear ---

    async def test_clear(
        self,
        monkeypatch,
        caplog,
        storage,
        mocked_chroma_client,
        mocked_chroma_collection,
        mocked_chroma_vector_store,
        mocked_storage_context,
        mocked_vector_store_index,
        mocked_huggingface_embedding,
    ):
        # mocked_chroma_client.delete_collection = MagicMock()

        mocked_chroma_client.create_collection.return_value = mocked_chroma_collection
        mock_chroma_vector_store_factory = MagicMock(return_value=mocked_chroma_vector_store)
        monkeypatch.setattr("notes_rag.storage.ChromaVectorStore", mock_chroma_vector_store_factory)
        mock_storage_context_from_defaults = MagicMock(return_value=mocked_storage_context)
        monkeypatch.setattr("notes_rag.storage.StorageContext.from_defaults", mock_storage_context_from_defaults)
        mock_vector_store_index_factory = MagicMock(return_value=mocked_vector_store_index)
        monkeypatch.setattr("notes_rag.storage.VectorStoreIndex", mock_vector_store_index_factory)
        mock_huggingface_embedding_factory = MagicMock(return_value=mocked_huggingface_embedding)
        monkeypatch.setattr("notes_rag.storage.HuggingFaceEmbedding", mock_huggingface_embedding_factory)

        with caplog.at_level(logging.INFO):
            await storage.clear()

        assert f"Cleared collection {storage.COLLECTION_NAME}" in caplog.text
        mocked_chroma_client.delete_collection.assert_called_once_with(storage.COLLECTION_NAME)
        mocked_chroma_client.create_collection.assert_called_once_with(name=storage.COLLECTION_NAME)
        mock_chroma_vector_store_factory.assert_called_once_with(chroma_collection=mocked_chroma_collection)
        mock_storage_context_from_defaults.assert_called_once_with(vector_store=mocked_chroma_vector_store)
        mock_huggingface_embedding_factory.assert_called_once_with(model_name=storage._config.embedding_model)
        mock_vector_store_index_factory.assert_called_once_with(
            use_async=True, storage_context=mocked_storage_context, embed_model=mocked_huggingface_embedding
        )

    # --- rebuild ---

    async def test_rebuild(self, caplog, storage, mocked_md_extractor):
        nodes = [
            TextNode(text="Node 1 content about machine learning", metadata={"source_file": "test1.md"}),
            TextNode(text="Node 2 content about neural networks", metadata={"source_file": "test2.md"}),
            TextNode(text="Node 3 content about neural networks", metadata={"source_file": "test2.md"}),
        ]
        mocked_md_extractor.get_nodes.return_value = ExtractorResult(
            files_processed=2,
            nodes=nodes,
        )
        storage.clear = AsyncMock()
        storage.add_nodes = AsyncMock()

        with caplog.at_level(logging.INFO):
            out = await storage.rebuild()

        assert out == 2
        storage.clear.assert_called_once()
        mocked_md_extractor.get_nodes.assert_called_once()
        storage.add_nodes.assert_called_once_with(nodes)
        assert "Rebuild complete: 2 files, 3 nodes indexed" in caplog.text

    async def test_rebuild_no_nodes(self, caplog, storage, mocked_md_extractor):
        mocked_md_extractor.get_nodes.return_value = ExtractorResult(
            files_processed=1,
            nodes=[],
        )
        storage.clear = AsyncMock()
        storage.add_nodes = AsyncMock()

        with caplog.at_level(logging.WARNING):
            out = await storage.rebuild()

        assert out == 1  # still, files_processed is 1
        storage.clear.assert_called_once()
        mocked_md_extractor.get_nodes.assert_called_once()
        storage.add_nodes.assert_not_called()
        assert "No nodes to be indexed!" in caplog.text

    async def test_rebuild_no_extractors(self, caplog, storage):
        storage._extractors = []
        storage.clear = AsyncMock()
        storage.add_nodes = AsyncMock()

        with caplog.at_level(logging.WARNING):
            out = await storage.rebuild()

        assert out == 0
        storage.clear.assert_called_once()
        storage.add_nodes.assert_not_called()
        assert "No nodes to be indexed!" in caplog.text
