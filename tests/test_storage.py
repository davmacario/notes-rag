import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

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
from notes_rag.storage import Storage


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
            chroma_collection=mocked_chroma_collection,
        )
        assert storage._vector_store == mocked_chroma_vector_store
        mock_storage_context_from_defaults.assert_called_once_with(vector_store=mocked_chroma_vector_store)
        assert storage._storage_context == mocked_storage_context
        mock_huggingface_embedding_factory.assert_called_once_with(model_name=mock_config.embedding_model)
        assert storage._embed_model == mocked_huggingface_embedding
        mock_vector_store_index_factory.assert_called_once_with(
            nodes=[], use_async=True, storage_context=mocked_storage_context, embed_model=mocked_huggingface_embedding
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
        mock_huggingface_embedding_factory.assert_not_called()  # Reusing existing one - class attribute
        mock_vector_store_index_factory.assert_called_once_with(
            nodes=[], use_async=True, storage_context=mocked_storage_context, embed_model=mocked_huggingface_embedding
        )

    # --- rebuild ---

    async def test_rebuild(
        self,
        monkeypatch,
        caplog,
        storage,
        mocked_md_extractor,
        mocked_chroma_client,
        mocked_chroma_collection,
        mocked_chroma_vector_store,
        mocked_storage_context,
        mocked_vector_store_index,
        mocked_huggingface_embedding,
    ):
        nodes = [
            TextNode(text="Chunk 1", metadata={"source_file": "doc1.md"}),
            TextNode(text="Chunk 2", metadata={"source_file": "doc1.md"}),
        ]
        extractor_result = ExtractorResult(nodes=nodes, files_processed=1)
        mocked_md_extractor.get_nodes.return_value = extractor_result

        tmp_collection = MagicMock(spec=Collection)
        call_count = [0]

        def get_or_create_side_effect(name):
            call_count[0] += 1
            if name == f"tmp_{storage.COLLECTION_NAME}":
                return tmp_collection
            return mocked_chroma_collection

        mocked_chroma_client.get_or_create_collection.side_effect = get_or_create_side_effect

        mock_chroma_vector_store_factory = MagicMock(return_value=mocked_chroma_vector_store)
        monkeypatch.setattr("notes_rag.storage.ChromaVectorStore", mock_chroma_vector_store_factory)
        mock_storage_context_from_defaults = MagicMock(return_value=mocked_storage_context)
        monkeypatch.setattr("notes_rag.storage.StorageContext.from_defaults", mock_storage_context_from_defaults)
        mock_vector_store_index_factory = MagicMock(return_value=mocked_vector_store_index)
        monkeypatch.setattr("notes_rag.storage.VectorStoreIndex", mock_vector_store_index_factory)

        mock_copy = MagicMock(return_value=2)
        monkeypatch.setattr("notes_rag.storage.copy_chroma_collection", mock_copy)

        with caplog.at_level(logging.INFO):
            result = await storage.rebuild()

        assert result == 1
        mocked_md_extractor.get_nodes.assert_called_once()
        mocked_vector_store_index.ainsert_nodes.assert_called_once_with(nodes)
        mocked_chroma_client.delete_collection.assert_called_with(storage.COLLECTION_NAME)
        mock_chroma_vector_store_factory.assert_has_calls(
            [call(chroma_collection=tmp_collection), call(chroma_collection=mocked_chroma_collection)]
        )
        mock_chroma_vector_store_factory.assert_called_with(chroma_collection=mocked_chroma_collection)
        mock_storage_context_from_defaults.assert_called_with(vector_store=mocked_chroma_vector_store)
        mock_vector_store_index_factory.assert_called_with(
            nodes=[], use_async=True, storage_context=mocked_storage_context, embed_model=mocked_huggingface_embedding
        )
        assert "Rebuild complete: 1 files, 2 nodes indexed" in caplog.text

    async def test_rebuild_no_nodes(
        self,
        monkeypatch,
        caplog,
        storage,
        mocked_md_extractor,
        mocked_chroma_client,
        mocked_chroma_collection,
        mocked_chroma_vector_store,
        mocked_storage_context,
        mocked_vector_store_index,
    ):
        extractor_result = ExtractorResult(nodes=[], files_processed=0)
        mocked_md_extractor.get_nodes.return_value = extractor_result

        tmp_collection = MagicMock(spec=Collection)
        mocked_chroma_client.get_or_create_collection.side_effect = [
            mocked_chroma_collection,
            tmp_collection,
            mocked_chroma_collection,
        ]
        mock_chroma_vector_store_factory = MagicMock(return_value=mocked_chroma_vector_store)
        monkeypatch.setattr("notes_rag.storage.ChromaVectorStore", mock_chroma_vector_store_factory)
        mock_storage_context_from_defaults = MagicMock(return_value=mocked_storage_context)
        monkeypatch.setattr("notes_rag.storage.StorageContext.from_defaults", mock_storage_context_from_defaults)
        mock_vector_store_index_factory = MagicMock(return_value=mocked_vector_store_index)
        monkeypatch.setattr("notes_rag.storage.VectorStoreIndex", mock_vector_store_index_factory)

        mock_copy = MagicMock(return_value=0)
        monkeypatch.setattr("notes_rag.storage.copy_chroma_collection", mock_copy)

        with caplog.at_level(logging.WARNING):
            result = await storage.rebuild()

        assert result == 0
        mocked_md_extractor.get_nodes.assert_called_once()
        mocked_vector_store_index.ainsert_nodes.assert_not_called()
        assert "No nodes to be indexed!" in caplog.text

    async def test_rebuild_no_extractors(
        self,
        monkeypatch,
        caplog,
        mock_config,
        mocked_chroma_client,
        mocked_chroma_collection,
        mocked_chroma_vector_store,
        mocked_storage_context,
        mocked_vector_store_index,
        mocked_huggingface_embedding,
    ):
        monkeypatch.setattr("notes_rag.storage.create_chroma_client", lambda *_: mocked_chroma_client)
        monkeypatch.setattr("notes_rag.storage.ChromaVectorStore", lambda **_: mocked_chroma_vector_store)
        monkeypatch.setattr("notes_rag.storage.StorageContext.from_defaults", lambda **_: mocked_storage_context)
        monkeypatch.setattr("notes_rag.storage.VectorStoreIndex", lambda **_: mocked_vector_store_index)
        monkeypatch.setattr("notes_rag.storage.HuggingFaceEmbedding", lambda **_: mocked_huggingface_embedding)

        storage_no_extractors = Storage(mock_config, [])

        tmp_collection = MagicMock(spec=Collection)
        call_count = [0]

        def get_or_create_side_effect(name=None, **kwargs):
            call_count[0] += 1
            actual_name = name or (kwargs.get("name") if kwargs else None)
            if actual_name == f"tmp_{storage_no_extractors.COLLECTION_NAME}":
                return tmp_collection
            return mocked_chroma_collection

        mocked_chroma_client.get_or_create_collection.side_effect = get_or_create_side_effect

        mock_copy = MagicMock(return_value=0)
        monkeypatch.setattr("notes_rag.storage.copy_chroma_collection", mock_copy)

        with caplog.at_level(logging.WARNING):
            result = await storage_no_extractors.rebuild()

        assert result == 0
        assert "No nodes to be indexed!" in caplog.text

    async def test_rebuild_multiple_extractors(
        self,
        monkeypatch,
        caplog,
        mock_config,
        storage,
        mocked_chroma_client,
        mocked_chroma_collection,
        mocked_chroma_vector_store,
        mocked_storage_context,
        mocked_vector_store_index,
        mocked_huggingface_embedding,
    ):
        nodes_a = [TextNode(text="A1", metadata={"source_file": "a.md"})]
        nodes_b = [
            TextNode(text="B1", metadata={"source_file": "b.md"}),
            TextNode(text="B2", metadata={"source_file": "b.md"}),
        ]
        extractor_result_a = ExtractorResult(nodes=nodes_a, files_processed=1)
        extractor_result_b = ExtractorResult(nodes=nodes_b, files_processed=2)
        ext_a = MagicMock(spec=MarkdownExtractor)
        ext_a.get_nodes.return_value = extractor_result_a
        ext_b = MagicMock(spec=MarkdownExtractor)
        ext_b.get_nodes.return_value = extractor_result_b

        storage._extractors = [ext_a, ext_b]

        tmp_collection = MagicMock(spec=Collection)
        call_count = [0]

        def get_or_create_side_effect(name=None, **kwargs):
            call_count[0] += 1
            actual_name = name or (kwargs.get("name") if kwargs else None)
            if actual_name == f"tmp_{storage.COLLECTION_NAME}":
                return tmp_collection
            return mocked_chroma_collection

        mocked_chroma_client.get_or_create_collection.side_effect = get_or_create_side_effect

        mock_chroma_vector_store_factory = MagicMock(return_value=mocked_chroma_vector_store)
        monkeypatch.setattr("notes_rag.storage.ChromaVectorStore", mock_chroma_vector_store_factory)
        mock_storage_context_from_defaults = MagicMock(return_value=mocked_storage_context)
        monkeypatch.setattr("notes_rag.storage.StorageContext.from_defaults", mock_storage_context_from_defaults)
        mock_vector_store_index_factory = MagicMock(return_value=mocked_vector_store_index)
        monkeypatch.setattr("notes_rag.storage.VectorStoreIndex", mock_vector_store_index_factory)

        mock_copy = MagicMock(return_value=3)
        monkeypatch.setattr("notes_rag.storage.copy_chroma_collection", mock_copy)

        with caplog.at_level(logging.INFO):
            result = await storage.rebuild()

        assert result == 3
        assert ext_a.get_nodes.call_count == 1
        assert ext_b.get_nodes.call_count == 1
        assert mocked_vector_store_index.ainsert_nodes.call_count == 2
        assert "Rebuild complete: 3 files, 3 nodes indexed" in caplog.text
