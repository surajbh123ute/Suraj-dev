from abc import ABC, abstractmethod
from llama_index.core import Document
from typing import List, Union

class BaseDataProcessor(ABC):
    """Abstract base class for processing raw data into LlamaIndex Documents."""
    @abstractmethod
    def load_data(self, input_source: Union[List, str]) -> List[Document]:
        pass