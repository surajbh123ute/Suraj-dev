from abc import ABC, abstractmethod
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from typing import List

class APIClient(ABC):
    """Abstract base class for API clients."""
    @abstractmethod
    def invoke(self, endpoint: str, headers: dict, payload: dict):
        pass

class ImageProcessor(ABC):
    """Abstract base class for image processing tasks."""
    @abstractmethod
    def get_base64_from_content(self, image_content: bytes) -> str:
        pass

class TextProcessor(ABC):
    """Abstract base class for text processing tasks."""
    @abstractmethod
    def process_text(self, text_blocks: list, **kwargs) -> list:
        pass

class FileHandler(ABC):
    """Abstract base class for file handling operations."""
    @abstractmethod
    def save_file(self, file_data, file_name: str, destination_path: str) -> str:
        pass

class BaseDocumentLoader(ABC):
    """Abstract base class for loading different document types."""
    @abstractmethod
    def load_documents(self, file) -> List[Document]:
        pass

class BaseDirectoryLoader(ABC):
    """Abstract base class for loading documents from a directory."""
    @abstractmethod
    def load_from_directory(self, directory: str) -> List[Document]:
        pass

class BaseEmbedding(ABC):
    """Abstract base class for embeddings."""
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

class BaseLLM(ABC):
    """Abstract base class for Language Model."""
    @abstractmethod
    def complete(self, prompt: str, **kwargs):
        pass

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

class BaseDataProcessor(ABC):
    """Abstract base class for data processors."""
    @abstractmethod
    def load_data(self, input_source) -> List[Document]:
        pass