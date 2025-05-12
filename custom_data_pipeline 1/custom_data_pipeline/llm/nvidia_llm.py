from typing import Any, Optional, Sequence
from llama_index.llms.nvidia import NVIDIA as LlamaNVIDIA
from llm.base_llm import BaseLLM
from config.settings import Settings

class NVIDIA(BaseLLM):
    """Wrapper around LlamaIndex's NVIDIA LLM."""
    def __init__(self, model: str = Settings.llm.model):
        self._llm = LlamaNVIDIA(model=model)

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        return self._llm.complete(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> Any:
        return await self._llm.acomplete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        return self._llm.stream_complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> Any:
        return await self._llm.astream_complete(prompt, **kwargs)

    @property
    def metadata(self) -> Any:
        return self._llm.metadata