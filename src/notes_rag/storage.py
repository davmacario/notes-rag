import asyncio
import logging
from pathlib import Path
from typing import List

from chromadb import Collection
from chromadb.errors import NotFoundError
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from notes_rag.config import Config
from notes_rag.extractor.abstract import BaseExtractor
from notes_rag.sub.storage_utils import create_chroma_client, prune_orphan_segments

logger = logging.getLogger(__name__)


class Storage:
    """ChromaDB storage wrapper for document embeddings.

    Provides an interface for adding, searching, and clearing documents in a ChromaDB collection. All documents are
    stored with metadata for tracking source files and chunk indices.

    This component is a Database Abstraction Layer
    """

    COLLECTION_NAME = "notes"
    TMP_COLLECTION_NAME = "tmp_notes"

    def __init__(
        self,
        config: Config,
        extractors: List[BaseExtractor] = [],
    ) -> None:
        """Initialize the ChromaDB storage.

        Args:
            config:
            extractors: List of Extractor objects for document ingestion.
            chroma_path: Path to ChromaDB storage directory
        """
        self._config = config
        self._extractors = extractors
        self._lock = asyncio.Lock()

        self._client = create_chroma_client(self.chroma_path)
        # Create (or get existing) ChromaDB collection
        self._chroma_collection: Collection = self._client.get_or_create_collection(name=self.COLLECTION_NAME)
        # Embedding model
        self._embed_model = FastEmbedEmbedding(model_name=self.embed_model)
        self._reset_llamaindex()
        logger.info(f"Initialized ChromaDB at {str(self.chroma_path)!r}")

    @property
    def embed_model(self) -> str:
        return self._config.embedding_model

    @property
    def chroma_path(self) -> Path:
        return self._config.chroma_path

    def _reset_llamaindex(self):
        """Recreate vector store, storage context, and index, all referencing the persistent Chroma collection"""
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex(
            nodes=[],
            use_async=True,
            storage_context=self._storage_context,
            embed_model=self._embed_model,
        )

    def _drop_collection_if_exists(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except NotFoundError:
            logger.debug(f"Collection {name!r} not found - unable to drop")

    async def search(self, query: str, top_k: int = 5) -> List[TextNode]:
        """Search for similar documents.

        Args:
            query: Query text to search for.
            top_k: Number of results to return.

        Returns:
            List of similar document nodes, sorted by similarity.
        """
        stripped_query = query.strip()
        if not stripped_query:
            return []

        # Retrieve: get top_k similar docs
        async with self._lock:
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            nodes_with_score = await retriever.aretrieve(query)

        logger.debug(f"Queried {top_k} nodes for query: {stripped_query!r}")

        nodes = [n.node for n in nodes_with_score]

        return nodes

    async def add_nodes(self, nodes: List[TextNode]) -> None:
        """Add TextNodes to the vector index.

        Uses VectorStoreIndex.insert() to generate embeddings and store them
        in ChromaDB.

        Args:
            nodes: List of TextNode objects to embed and store.
        """
        if not nodes:
            return

        async with self._lock:
            # Insert nodes into the index, which generates embeddings via the embed_model passed at construction time
            await self._index.ainsert_nodes(nodes)
            logger.info(f"Inserted {len(nodes)} documents with embeddings")

    async def clear(self) -> None:
        """Clear all documents from the collection."""
        async with self._lock:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._chroma_collection = self._client.create_collection(name=self.COLLECTION_NAME)
            self._reset_llamaindex()
            logger.info(f"Cleared collection {self.COLLECTION_NAME}")

    async def rebuild(self) -> None:
        """Rebuild the vector index by iterating over all extractors.

        Atomic rebuild: first builds new as tmp collection, then swaps it to the live one to minimize downtime.
        """
        logger.info("Rebuilding vector DB")

        # Always start from an empty collection (previous runs can leave leftovers)
        self._drop_collection_if_exists(self.TMP_COLLECTION_NAME)
        tmp_collection = self._client.create_collection(name=self.TMP_COLLECTION_NAME)
        swapped = False
        try:
            tmp_vector_store = ChromaVectorStore(chroma_collection=tmp_collection)
            tmp_storage_context = StorageContext.from_defaults(vector_store=tmp_vector_store)
            tmp_index = VectorStoreIndex(
                nodes=[],
                use_async=True,
                storage_context=tmp_storage_context,
                embed_model=self._embed_model,
            )

            nodes_count = 0
            for extractor in self._extractors:
                for nodes_batch in extractor.get_nodes(batch_size=400):
                    await tmp_index.ainsert_nodes(nodes_batch)
                    logger.debug(f"Added {len(nodes_batch)} nodes to vector db")
                    nodes_count += len(nodes_batch)

            if nodes_count:
                logger.info(f"Rebuild complete: {nodes_count} nodes indexed")
            else:
                logger.warning("No nodes to be indexed!")

            # swap live collection
            async with self._lock:
                self._drop_collection_if_exists(self.COLLECTION_NAME)
                tmp_collection.modify(name=self.COLLECTION_NAME)
                swapped = True
                self._chroma_collection = self._client.get_collection(name=self.COLLECTION_NAME)
                self._reset_llamaindex()
        finally:
            # Safely exit - allows to always prune something to avoid it being left behind
            if not swapped:
                self._drop_collection_if_exists(self.TMP_COLLECTION_NAME)
            removed = await asyncio.to_thread(prune_orphan_segments, self.chroma_path)
            if removed:
                logger.info(f"Pruned {len(removed)} orphan segment directories")
