import os
from llama_index.llms.nvidia import NVIDIA
from clients.nvidia_client import NVIDIAAPIClient
from processors.image_processor import PILImageProcessor
from processors.text_processor import FitzTextProcessor
from llama_index.core import Document
from document_loaders.pdf_loader import PDFDocumentLoader
from document_loaders.ppt_loader import PPTXDocumentLoader
from document_loaders.image_loader import ImageDocumentLoader
from document_loaders.text_loader import TextDocumentLoader
from core.enums import FileType
from utils.file_utils import save_uploaded_file
from typing import List

def set_environment_variables():
    """Set necessary environment variables."""
    os.environ["NVIDIA_API_KEY"] = "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" #set API key

def process_graph(image_content: bytes):
    """Process a graph image and generate a description."""
    nvidia_client = NVIDIAAPIClient()
    image_processor = PILImageProcessor()
    deplot_description = nvidia_client.process_graph_deplot(image_processor.get_base64_from_content(image_content))
    mixtral = NVIDIA(model_name="mistralai/mixtral-8x7b-instruct-v0.1")
    response = mixtral.complete(f"Your responsibility is to explain charts. You are an expert in describing the responses of linearized tables into plain English text for LLMs to use. Explain the following linearized table. {deplot_description}")
    return response.text

def describe_image(image_content: bytes) -> str:
    """Generate a description of an image using NVIDIA API."""
    nvidia_client = NVIDIAAPIClient()
    image_processor = PILImageProcessor()
    image_b64 = image_processor.get_base64_from_content(image_content)
    return nvidia_client.describe_image(image_b64)

def is_graph(image_content: bytes) -> bool:
    """Determine if an image is a graph, plot, chart, or table."""
    image_processor = PILImageProcessor()
    return image_processor.is_graph(image_content, describe_image)

def extract_text_around_item(text_blocks: list, bbox, page_height: float, threshold_percentage: float = 0.1) -> tuple[str, str]:
    """Extract text above and below a given bounding box on a page."""
    text_processor = FitzTextProcessor()
    return text_processor.extract_text_around_item(text_blocks, bbox, page_height, threshold_percentage)

def process_text_blocks(text_blocks: list, char_count_threshold: int = 500) -> list[tuple[tuple, str]]:
    """Group text blocks based on a character count threshold."""
    text_processor = FitzTextProcessor()
    return text_processor.process_text(text_blocks, char_count_threshold)

# Note: save_uploaded_file is now in utils/file_handler.py
from utils.file_handler import save_uploaded_file

def load_multimodel_data(files) -> List[Document]:
    """Load and process multiple file types."""
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
                # Re-open the saved file for the loader
                with open(uploaded_path, 'rb') as ppt_saved_file:
                    documents.extend(pptx_loader.load_documents(ppt_saved_file))
                os.remove(uploaded_path) # Clean up the temporary file
            elif file_extension == FileType.TEXT.value:
                documents.extend(text_loader.load_documents(file))
            else:
                print(f"Unsupported file type: {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
    return documents

def load_data_from_directory(directory: str) -> List[Document]:
    """Load and process multiple file types from a directory."""
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
                    # Re-open in text mode for text loader
                    with open(filepath, 'r', encoding='utf-8') as text_f:
                        documents.extend(text_loader.load_documents(text_f))
                else:
                    print(f"Unsupported file type: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return documents

if __name__ == "__main__":
    # Example usage:
    # Assuming you have files in a list called 'uploaded_files'
    # loaded_docs = load_multimodel_data(uploaded_files)
    # for doc in loaded_docs:
    #     print(f"Source: {doc.metadata['source']}, Type: {doc.metadata['type']}, Text: {doc.text[:50]}...")

    # Assuming you have a directory called 'data_directory'
    # directory_docs = load_data_from_directory('data_directory')
    # for doc in directory_docs:
    #     print(f"Source: {doc.metadata['source']}, Type: {doc.metadata['type']}, Text: {doc.text[:50]}...")
    pass

# if __name__ == "__main__":
    # set_environment_variables()
    # Example usage (you would need to provide an actual image file)
    # with open("your_image.jpg", "rb") as f:
    #     image_data = f.read()
    #     if is_graph(image_data):
    #         description = process_graph(image_data)
    #         print("Graph Description:", description)
    #     else:
    #         description = describe_image(image_data)
    #         print("Image Description:", description)