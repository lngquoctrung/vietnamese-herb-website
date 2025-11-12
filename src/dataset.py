import os
import pdf2image

from qdrant_client import QdrantClient
from .utils import get_logger

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.qdrant_client = QdrantClient()
    
    def _pdf_to_images(self, pdf_path, dpi=300):

        os.makedirs(self.config.IMAGE_DATA_PATH, exist_ok=True)
        
        images = pdf2image.convert_from_path(
            pdf_path=pdf_path,
            dpi=dpi,
            output_folder=self.config.IMAGE_DATA_PATH,
            fmt="png"
        )
        return images

    def _extract_text_from_pdf(self):
        pass

    def extract_content_pdf(self, filepath: str):
        images = self._pdf_to_images(pdf_path=filepath)
        return images