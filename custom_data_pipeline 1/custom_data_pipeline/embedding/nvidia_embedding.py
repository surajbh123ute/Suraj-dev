from typing import List
from llama_index.embeddings.nvidia import NVIDIAEmbedding as LlamaNVIDIAEmbedding
from embedding.base_embedding import BaseEmbedding
from config.settings import Settings

class NVIDIAEmbedding(BaseEmbedding):
    """Wrapper around LlamaIndex's NVIDIAEmbedding."""
    def __init__(self, model: str = Settings.embed_model.model, truncate: str = Settings.embed_model.truncate):
        self._embedding_model = LlamaNVIDIAEmbedding(model=model, truncate=truncate)

    def get_embedding(self, text: str) -> List[float]:
        return self._embedding_model.get_embedding(text)

    async def aget_embedding(self, text: str) -> List[float]:
        return await self._embedding_model.aget_embedding(text)

    @property
    def model_name(self) -> str:
        return self._embedding_model.model_name

    @property
    def embed_batch_size(self) -> int:
        return self._embedding_model.embed_batch_size

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_model._get_embedding(texts)