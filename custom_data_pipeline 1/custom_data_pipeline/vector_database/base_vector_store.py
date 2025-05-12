from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Document

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    @abstractmethod
    def add(
        self,
        keys: List[str],
        embeddings: List[List[float]],
        nodes: List[Document],
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def query(self, query: List[float], k: int = 1, **kwargs) -> List[Document]:
        pass