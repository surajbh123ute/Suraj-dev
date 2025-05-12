import os
from llama_index.core import Document
from data_processing.base_processor import BaseDataProcessor
from document_loaders.pdf_loader import PDFDocumentLoader
from document_loaders.ppt_loader import PPTXDocumentLoader
from document_loaders.image_loader import ImageDocumentLoader
from document_loaders.text_loader import TextDocumentLoader
from core.enums import FileType
from utils.file_utils import save_uploaded_file
from typing import List

class MultimodalDocumentLoader(BaseDataProcessor):
    """Loads data from various file types."""
    def load_data(self, files: List) -> List[Document]:
        documents = []
        pdf_loader = PDFDocumentLoader()
        pptx_loader = PPTXDocumentLoader()
        image_loader = ImageDocumentLoader()
        text_loader = TextDocumentLoader()

        for file in files:
            file_extension = os.path.splitext(file.name.lower())[1]
            try:
                if file_extension == FileType.PNG.value or file_extension == FileType.JPG.value or file_extension == FileType.JPEG.value:
                    documents.extend(image_loader.load_documents(file))
                elif file_extension == FileType.PDF.value:
                    documents.extend(pdf_loader.load_documents(file))
                elif file_extension == FileType.PPT.value or file_extension == FileType.PPTX.value:
                    uploaded_path = save_uploaded_file(file)
                    with open(uploaded_path, 'rb') as ppt_saved_file:
                        documents.extend(pptx_loader.load_documents(ppt_saved_file))
                    os.remove(uploaded_path)
                elif file_extension == FileType.TEXT.value:
                    documents.extend(text_loader.load_documents(file))
                else:
                    print(f"Unsupported file type: {file.name}")
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
        return documents

class DirectoryDocumentLoader(BaseDataProcessor):
    """Loads data from a directory of various file types."""
    def load_data(self, directory: str) -> List[Document]:
        documents = []
        pdf_loader = PDFDocumentLoader()
        pptx_loader = PPTXDocumentLoader()
        image_loader = ImageDocumentLoader()
        text_loader = TextDocumentLoader()

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            file_extension = os.path.splitext(filename.lower())[1]
            try:
                with open(filepath, 'rb') as f:
                    if file_extension == FileType.PNG.value or file_extension == FileType.JPG.value or file_extension == FileType.JPEG.value:
                        documents.extend(image_loader.load_documents(f))
                    elif file_extension == FileType.PDF.value:
                        documents.extend(pdf_loader.load_documents(f))
                    elif file_extension == FileType.PPT.value or file_extension == FileType.PPTX.value:
                        documents.extend(pptx_loader.load_documents(f))
                    elif file_extension == FileType.TEXT.value:
                        with open(filepath, 'r', encoding='utf-8') as text_f:
                            documents.extend(text_loader.load_documents(text_f))
                    else:
                        print(f"Unsupported file type: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        return documents

load_multimodel_data = MultimodalDocumentLoader().load_data
load_data_from_directory = DirectoryDocumentLoader().load_data