from abc import ABC, abstractmethod
from typing import List

class BaseEmbedding(ABC):
    """Abstract base class for embeddings."""
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def aget_embedding(self, text: str) -> List[float]:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def embed_batch_size(self) -> int:
        pass

    @abstractmethod
    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        pass

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        return self._get_embedding(texts)

    async def aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts asynchronously."""
        return self._get_embedding(texts) # Assuming synchronous for base class