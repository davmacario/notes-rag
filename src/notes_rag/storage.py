import asyncio
import logging
from pathlib import Path
from typing import List

import chromadb.config
from chromadb import Collection, PersistentClient
from chromadb.api import ClientAPI
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from notes_rag.config import Config
from notes_rag.extractor.abstract import BaseExtractor

logger = logging.getLogger(__name__)


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


class Storage:
    """ChromaDB storage wrapper for document embeddings.

    Provides an interface for adding, searching, and clearing documents in a ChromaDB collection. All documents are
    stored with metadata for tracking source files and chunk indices.

    This component is a Database Abstraction Layer
    """

    COLLECTION_NAME = "notes"

    def __init__(
        self,
        config: Config,
        extractors: List[BaseExtractor] = [],
        chroma_path: str | Path | None = None,
    ) -> None:
        """Initialize the ChromaDB storage.

        Args:
            extractors: List of Extractor objects for document ingestion.
            chroma_path: Path to ChromaDB storage directory
        """
        self.config = config

        self._extractors = extractors

        # TODO: actually use lock
        self._lock = asyncio.Lock()

        if chroma_path:
            self.chroma_path = Path(chroma_path)
        else:
            self.chroma_path = Path("./.chromadb")

        self._client = create_chroma_client(self.chroma_path)

        # Create (or get existing) ChromaDB collection
        self._chroma_collection: Collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        # Create vector store wrapper and storage context
        self._vector_store = ChromaVectorStore(
            chroma_collection=self._chroma_collection, collection_name=self.COLLECTION_NAME
        )
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

        # Create an empty VectorStoreIndex that will be populated via insert()
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=self._storage_context,
            embed_model=HuggingFaceEmbedding(model_name=self.embed_model),
        )
        logger.info(f"Initialized ChromaDB at {str(self.chroma_path)!r}")

    @property
    def embed_model(self) -> str:
        """Get the embedding model name from configuration."""
        return self.config.embedding_model

    @property
    def _collection(self):
        """Get the ChromaDB collection instance."""
        return self._client.get_collection(name=self.COLLECTION_NAME)

    async def search(self, query: str, top_k: int = 5) -> List[BaseNode]:
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
            nodes_with_score = retriever.retrieve(query)

        logger.debug(f"Queried {top_k} nodes for query: {stripped_query!r}")

        # TODO: figure out how to inject into the context
        nodes = [n.node for n in nodes_with_score]

        return nodes

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """Add TextNodes to the vector index.

        Uses VectorStoreIndex.insert() to generate embeddings and store them
        in ChromaDB.

        Args:
            nodes: List of TextNode objects to embed and store.
        """
        if not nodes:
            return

        # Insert nodes into the index, which generates embeddings via the embed_model passed at construction time
        self._index.insert_nodes(nodes)
        logger.info(f"Inserted {len(nodes)} documents with embeddings")

    # TODO: define logic to minimize downtime (rework `clear` and `rebuild`)
    # Ideally, a tmp DB is created and then it is swapped to the actual one.
    # This will require using a lock for DB-related operations.

    def clear(self) -> None:
        """Clear all documents from the collection.

        This drops the collection and recreates it, along with a fresh index.
        """
        # TODO: unify logic with __init__
        self._client.delete_collection(self.COLLECTION_NAME)
        self._chroma_collection = self._client.create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        # Recreate vector store, storage context, and index
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=self._storage_context,
            embed_model=HuggingFaceEmbedding(model_name=self.embed_model),
        )
        logger.info(f"Cleared collection {self.COLLECTION_NAME}")

    def rebuild(self) -> int:
        """Rebuild the vector index by iterating over all extractors.

        Clears the collection, calls each extractor's rebuild() method,
        and stores the returned nodes in ChromaDB with embeddings.

        Returns:
            number of files processed
        """
        self.clear()

        files_processed = 0
        all_nodes: List[TextNode] = []

        for extractor in self._extractors:
            extractor_result = extractor.get_nodes()
            files_processed += extractor_result.files_processed
            all_nodes.extend(extractor_result.nodes)

        if all_nodes:
            self.add_nodes(all_nodes)
            logger.info(f"Rebuild complete: {files_processed} files, {len(all_nodes)} nodes indexed")
        else:
            logger.warning("No nodes to be indexed!")

        return files_processed
