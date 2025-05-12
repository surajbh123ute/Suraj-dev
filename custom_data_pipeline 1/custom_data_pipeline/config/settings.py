from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA

def initialize_llama_settings():
    Settings.embed_model = NVIDIAEmbedding()
    Settings.llm = NVIDIA()
    Settings.text_splitter = SentenceSplitter(chunk_size=600)