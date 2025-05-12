import base64
from io import BytesIO
from PIL import Image
from core.base import ImageProcessor
from core.enums import ImageFormat

class PILImageProcessor(ImageProcessor):
    """Image processor using Pillow (PIL) library."""

    def get_base64_from_content(self, image_content: bytes) -> str:
        """Convert image content to base64 encoded string."""
        img = Image.open(BytesIO(image_content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format=ImageFormat.JPEG.value)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def is_graph(self, image_content: bytes, description_function) -> bool:
        """Determine if an image is a graph, plot, chart, or table."""
        res = description_function(image_content)
        return any(keyword in res.lower() for keyword in ["graph", "plot", "chart", "table"])