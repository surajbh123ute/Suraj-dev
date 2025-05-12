from typing import List
from llama_index.vector_stores.milvus import MilvusVectorStore as LlamaMilvus
from llama_index.core import Document, StorageContext
from vector_database.base_vector_store import BaseVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding

class MilvusVectorStore(BaseVectorStore):
    """Wrapper around LlamaIndex's MilvusVectorStore."""
    def __init__(self, host: str, port: int, dim: int):
        self._vector_store = LlamaMilvus(host=host, port=port, dim=dim)

    def add(
        self,
        keys: List[str],
        embeddings: List[List[float]],
        nodes: List[Document],
        **kwargs,
    ) -> None:
        self._vector_store.add(keys=keys, embeddings=embeddings, nodes=nodes, **kwargs)

    def query(self, query: List[float], k: int = 1, **kwargs) -> List[Document]:
        results = self._vector_store.query(query=query, k=k, **kwargs)
        return [node.as_document() for node in results.nodes]

    def create_index_from_documents(self, documents: List[Document]) -> StorageContext:
        """Creates a VectorStoreIndex from documents using the Milvus vector store."""
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return storage_context