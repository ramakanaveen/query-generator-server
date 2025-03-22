from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings
from pathlib import Path
import os
from google.cloud import aiplatform

class LLMProvider:
    """
    Service for providing language models from different providers.
    Supports Gemini (via VertexAI) and Claude (via Anthropic).
    """
    
    def __init__(self):
        self.models = {
            "gemini": self._init_gemini,
            "claude": self._init_claude
        }
    
    def _init_gemini(self):
        """Initialize and return a Gemini model instance."""
        # Get the credentials file path from settings
        credentials_path = settings.GOOGLE_CREDENTIALS_PATH
        
        # Check if the file exists
        credentials_file = Path(credentials_path)
        if not credentials_file.exists():
            raise FileNotFoundError(f"Google credentials file not found at {credentials_path}")

        # Set the environment variable for Google authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)
        
        # Initialize AI Platform with project and location
        aiplatform.init(
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION
        )
            
        return ChatVertexAI(
            model_name=settings.GEMINI_MODEL_NAME,
            temperature=settings.GEMINI_TEMPERATURE,
            top_p=settings.GEMINI_TOP_P,
            verbose=settings.DEBUG,
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
        )
    
    def _init_claude(self):
        """Initialize and return a Claude model instance."""
        return ChatAnthropic(
            model_name=settings.CLAUDE_MODEL_NAME,
            temperature=settings.CLAUDE_TEMPERATURE,
            anthropic_api_key=settings.API_KEY_ANTHROPIC,
            max_tokens=settings.CLAUDE_MAX_TOKENS,
        )
    
    def get_model(self, model_name="gemini"):
        """
        Get a language model by name.
        
        Args:
            model_name: Name of the model to use ("gemini" or "claude")
            
        Returns:
            A LangChain chat model instance
            
        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in self.models:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.models.keys())}")
        
        return self.models[model_name]()