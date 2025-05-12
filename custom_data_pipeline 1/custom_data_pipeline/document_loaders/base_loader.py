from core.base import BaseDocumentLoader
from abc import abstractmethod
from llama_index.core import Document
from core.enums import DocumentType
from typing import Optional, List

class SimpleDocumentLoader(BaseDocumentLoader):
    """Base class for simple file-based document loaders."""
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path

    @abstractmethod
    def _load_data(self, file) -> List[Document]:
        """Load data from the file."""
        pass

    def load_documents(self, file) -> List[Document]:
        """Load documents from the specified file."""
        return self._load_data(file)

class BaseMultiModalLoader(BaseDocumentLoader):
    """Base class for loaders that handle multiple data types within a file."""
    @abstractmethod
    def load_documents(self, file) -> List[Document]:
        pass