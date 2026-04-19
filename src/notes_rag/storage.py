import logging
from typing import List
from uuid import uuid4

from chromadb import Collection, PersistentClient
from chromadb.api import ClientAPI
from chromadb.config import Settings
from llama_index.core import Document, get_response_synthesizer
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from notes_rag.config import Config

logger = logging.getLogger(__name__)


def create_chroma_client(path: str) -> ClientAPI:
    """Create a ChromaDB persistent client.

    Args:
        path: Path to the ChromaDB storage directory.

    Returns:
        ChromaDB PersistentClient instance.
    """
    # Use anonymous mode to avoid metadata conflicts
    client = PersistentClient(
        path=path,
        settings=Settings(
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

    def __init__(self, chroma_path: str | None = None) -> None:
        """Initialize the ChromaDB storage.

        Args:
            chroma_path: Path to ChromaDB storage directory
        """
        if chroma_path:
            self.chroma_path = chroma_path
        else:
            config = Config()
            self.chroma_path = str(config.notes_cache_dir / ".chromadb")

        # Create ChromaDB client (possibly from existing path)
        self._client = create_chroma_client(self.chroma_path)

        # Create (or get existing) ChromaDB collection
        self._chroma_collection: Collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        # Create vector store wrapper
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        logger.info(f"Initialized ChromaDB at {self.chroma_path}")

    @property
    def _collection(self):
        """Get the ChromaDB collection instance."""
        return self._client.get_collection(name=self.COLLECTION_NAME)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the ChromaDB collection.

        Args:
            documents: List of LlamaIndex Document objects to embed and store. Each document should have metadata
                containing source file information.
        """
        # Convert to TextNodes
        nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]

        ids = [str(uuid4()) for _ in nodes]
        texts = [node.text for node in nodes]
        metadatas = [node.metadata for node in nodes]

        self._chroma_collection.add(ids=ids, documents=texts, metadatas=metadatas)

        logger.info(f"Added {len(documents)} documents to storage")

    def search(self, query: str, top_k: int = 5) -> List[BaseNode]:
        """Search for similar documents.

        Args:
            query: Query text to search for.
            top_k: Number of results to return.

        Returns:
            List of similar document nodes, sorted by similarity.
        """
        if not query.strip():
            return []

        # Use the vector store's query engine for search
        query_engine = self._vector_store.as_query_engine()
        response = query_engine.query(query)

        # Convert response sources to BaseNode objects
        nodes = []
        for source in response.source_nodes:
            nodes.append(BaseNode(node_id=source.node_id, text=source.text, metadata=source.metadata))

        return nodes

    def clear(self) -> None:
        """Clear all documents from the collection.

        This drops the collection and recreates it.
        """
        self._client.delete_collection(self.COLLECTION_NAME)
        self._chroma_collection = self._client.create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection {self.COLLECTION_NAME}")
