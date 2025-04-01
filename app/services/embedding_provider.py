# app/services/embedding_provider.py
import os
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings
from app.core.config import settings
from app.core.logging import logger

class EmbeddingProvider:
    """Service for generating embeddings using Google's text-embeddings-gecko model."""
    
    from langchain_google_vertexai import VertexAIEmbeddings

class EmbeddingProvider:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(settings.GOOGLE_CREDENTIALS_PATH)
        self.embeddings = VertexAIEmbeddings(
            model_name=settings.GOOGLE_EMBEDDING_MODEL_NAME,
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
        )
        
    async def get_embedding(self, text):
        return await self.embeddings.aembed_query(text)
        
        