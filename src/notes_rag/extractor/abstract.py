from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from llama_index.core.schema import TextNode


@dataclass
class ExtractorResult:
    nodes: List[TextNode]
    files_processed: int


class BaseExtractor(ABC):
    @abstractmethod
    def get_nodes(self) -> ExtractorResult: ...
