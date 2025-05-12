import os
from core.base import FileHandler

class LocalFileHandler(FileHandler):
    """File handler for local file system operations."""

    def save_file(self, file_data, file_name: str, destination_path: str) -> str:
        """Save data to a file in the specified destination path."""
        os.makedirs(destination_path, exist_ok=True)
        file_path = os.path.join(destination_path, file_name)
        try:
            with open(file_path, "wb") as f:
                f.write(file_data)
            return file_path
        except Exception as e:
            raise Exception(f"Error saving file to {file_path}: {e}")

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory."""
    file_handler = LocalFileHandler()
    temp_dir = os.path.join(os.getcwd(), "vectorstore", "ppt_references", "tmp")
    return file_handler.save_file(uploaded_file.read(), uploaded_file.name, temp_dir)