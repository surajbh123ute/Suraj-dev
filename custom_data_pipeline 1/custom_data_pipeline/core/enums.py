from enum import Enum

class NVIDIAModel(Enum):
    """Enum for NVIDIA LLM models."""
    NEVA_22B = "nvidia/neva-22b"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1" # Corrected model name
    MIXTRAL_8x7B = "mistralai/mixtral-8x7b-instruct-v0.1"

class NVIDIAEndpoint(Enum):
    """Enum for NVIDIA API endpoints."""
    VLM_BASE = "https://ai.api.nvidia.com/v1/vlm"
    DESCRIBE_IMAGE = f"{VLM_BASE}/nvidia/{NVIDIAModel.NEVA_22B.value}"
    DEPLOT = f"{VLM_BASE}/google/deplot"

class ImageFormat(Enum):
    """Enum for image formats."""
    JPEG = "JPEG"
    PNG = "PNG"

class TextBlockType(Enum):
    """Enum for text block types."""
    TEXT = 0

class LLMRole(Enum):
    """Enum for LLM message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class FileType(Enum):
    """Enum for different file types."""
    PDF = ".pdf"
    PPT = ".ppt"
    PPTX = ".pptx"
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    TEXT = ".txt"

class DocumentType(Enum):
    """Enum for the type of document being processed."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class TableParsingStrategy(Enum):
    """Enum for different table parsing strategies."""
    LINES_STRICT_HORIZONTAL = "lines_strict"
    LINES_STRICT_VERTICAL = "lines_strict"

class EmbeddingModel(Enum):
    """Enum for available embedding models."""
    NVIDIA_E5_V5 = "nvidia/nv-embedqa-e5-v5"

class LLMModel(Enum):
    """Enum for available Language Models."""
    LLAMA_3_70B_INSTRUCT = "meta/llama-3.1-70b-instruct"

class VectorStoreType(Enum):
    """Enum for available vector store types."""
    MILVUS = "milvus"

class InputMethod(Enum):
    """Enum for user input methods."""
    UPLOAD_FILES = "Upload Files"
    ENTER_DIRECTORY = "Enter Directory Path"
