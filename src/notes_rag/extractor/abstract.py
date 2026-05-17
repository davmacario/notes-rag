from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import List

from llama_index.core.schema import TextNode

@dataclass
class ExtractorResult:
    nodes: List[TextNode]
    files_processed: int

# FIXME: is Thread needed?
class BaseExtractor(ABC, Thread):
    @abstractmethod
    def get_nodes(self) -> ExtractorResult:
        pass
