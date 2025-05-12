import os
import requests
from core.base import APIClient
from core.enums import NVIDIAEndpoint

class NVIDIAAPIClient(APIClient):
    """Client for interacting with NVIDIA APIs."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def invoke(self, endpoint: str, payload: dict):
        """Invokes the NVIDIA API with the given endpoint and payload."""
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with NVIDIA API: {e}")

    def describe_image(self, image_b64: str) -> str:
        """Generates a description of an image using the NVIDIA API."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'Describe what you see in this image. <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.70,
            "seed": 0,
            "stream": False
        }
        response_data = self.invoke(NVIDIAEndpoint.DESCRIBE_IMAGE.value, payload)
        return response_data["choices"][0]['message']['content']

    def process_graph_deplot(self, image_b64: str) -> str:
        """Processes a graph image using NVIDIA's Deplot API."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.20,
            "stream": False
        }
        response_data = self.invoke(NVIDIAEndpoint.DEPLOT.value, payload)
        return response_data["choices"][0]['message']['content']