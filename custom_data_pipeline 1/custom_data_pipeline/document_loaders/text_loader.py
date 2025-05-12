from llama_index.core import Document
from document_loaders.base_loader import SimpleDocumentLoader
from core.enums import DocumentType

class TextDocumentLoader(SimpleDocumentLoader):
    """Loader for plain text files."""
    def _load_data(self, text_file) -> list[Document]:
        text = text_file.read().decode("utf-8")
        doc = Document(text=text, metadata={"source": text_file.name, "type": DocumentType.TEXT.value})
        return [doc]