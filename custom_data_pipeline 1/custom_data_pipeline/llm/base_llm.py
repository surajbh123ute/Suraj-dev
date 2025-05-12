from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

class BaseLLM(ABC):
    """Abstract base class for Language Model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model name."""
        pass

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Completes the given prompt."""
        pass

    @abstractmethod
    async def acomplete(self, prompt: str, **kwargs: Any) -> Any:
        """Completes the given prompt asynchronously."""
        pass

    @abstractmethod
    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        """Streams the completion of the given prompt."""
        pass

    @abstractmethod
    async def astream_complete(self, prompt: str, **kwargs: Any) -> Any:
        """Streams the completion of the given prompt asynchronously."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Any:
        """Returns the LLM metadata."""
        pass