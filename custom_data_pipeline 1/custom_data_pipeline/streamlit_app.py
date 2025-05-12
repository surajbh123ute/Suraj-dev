import streamlit as st
import os
from llama_index.core import VectorStoreIndex
from config.settings import initialize_llama_settings
from data_processing.document_loader import load_multimodel_data, load_data_from_directory
from vector_database.milvus_vector_store import MilvusVectorStore
from embedding.nvidia_embedding import NVIDIAEmbedding
from llm.nvidia_llm import NVIDIA
from core.enums import InputMethod

# Set up the page configuration
st.set_page_config(layout="wide")

# Initialize settings
initialize_llama_settings()

# Create index from documents
def create_index(documents):
    embedding_model = NVIDIAEmbedding()
    vector_store = MilvusVectorStore(
        host="127.0.0.1",
        port=8501,
        dim=embedding_model.embed_batch_size # Use embedding dimension
    )
    storage_context = vector_store.create_index_from_documents(documents)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Main function to run the Streamlit app