from llama_index.core import Document
from document_loaders.base_loader import SimpleDocumentLoader
from core.enums import DocumentType
from utils import describe_image

class ImageDocumentLoader(SimpleDocumentLoader):
    """Loader for image files (PNG, JPG, JPEG)."""
    def _load_data(self, image_file) -> list[Document]:
        image_content = image_file.read()
        image_text = describe_image(image_content)
        doc = Document(text=image_text, metadata={"source": image_file.name, "type": DocumentType.IMAGE.value})
        return [doc]