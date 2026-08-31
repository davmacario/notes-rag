from abc import ABC, abstractmethod
from typing import Generator, List

from llama_index.core.schema import TextNode


class BaseExtractor(ABC):
    @abstractmethod
    def get_nodes(self, batch_size: int | None = None) -> Generator[List[TextNode], None, None]:
        """Process files and yield batches of nodes for embedding by Storage.

        Walks through the notes directory recursively, parses files, chunks them,
        and yields lists of TextNodes. Storage will handle embedding and storing.

        Args:
            batch_size: number of nodes per yielded batch; if None, yields all nodes in a single batch.

        Yields:
            Lists of TextNode objects.

        Raises:
            FileNotFoundError: if the local repository copy is not found after cloning.
        """
        ...
