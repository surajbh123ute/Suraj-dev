import os

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory."""
    temp_dir = os.path.join(os.getcwd(), "vectorstore", "ppt_references", "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    return temp_file_path

__all__ = ['save_uploaded_file']