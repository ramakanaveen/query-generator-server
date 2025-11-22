# app/services/llm_provider.py
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings
from pathlib import Path
import os
from google.cloud import aiplatform
import time
from app.core.langfuse_client import langfuse_client # Import your Langfuse client

class LLMProvider:
    """
    Service for providing language models from different providers.
    Supports Gemini (via VertexAI) and Claude (via Anthropic).

    Implements a singleton pattern with lazy initialization and caching.
    """

    _instance = None
    _models = {}  # Cache for initialized models
    _last_use = {}  # Track last use time for each model
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMProvider, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._model_initializers = {
                "gemini": self._init_gemini,
                "gemini-fast": self._init_gemini_fast,
                "claude": self._init_claude
            }
            self._initialized = True

    def _init_gemini(self):
        """Initialize and return a Gemini model instance."""
        # Check if already initialized and recently used (within last 30 minutes)
        if "gemini" in self._models:
            return self._models["gemini"]

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

        # Create and cache the model
        model = ChatVertexAI(
            model_name=settings.GEMINI_MODEL_NAME,
            temperature=settings.GEMINI_TEMPERATURE,
            top_p=settings.GEMINI_TOP_P,
            verbose=settings.DEBUG,
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
        )
        # Attach Langfuse callback handler
        if langfuse_client.get_callback_handler():
            model.callbacks = [langfuse_client.get_callback_handler()] #

        self._models["gemini"] = model
        return model

    def _init_gemini_fast(self):
        """Initialize and return a Gemini Fast model instance."""
        # Check if already initialized
        if "gemini-fast" in self._models:
            return self._models["gemini-fast"]

        # Ensure credentials and init (re-using logic from _init_gemini would be better but keeping it simple)
        credentials_path = settings.GOOGLE_CREDENTIALS_PATH
        credentials_file = Path(credentials_path)
        if not credentials_file.exists():
             raise FileNotFoundError(f"Google credentials file not found at {credentials_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)
        
        # Initialize AI Platform (idempotent)
        aiplatform.init(
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION
        )

        # Create and cache the model
        model = ChatVertexAI(
            model_name=settings.GEMINI_FAST_MODEL_NAME,
            temperature=settings.GEMINI_TEMPERATURE,
            top_p=settings.GEMINI_TOP_P,
            verbose=settings.DEBUG,
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
        )
        # Attach Langfuse callback handler
        if langfuse_client.get_callback_handler():
            model.callbacks = [langfuse_client.get_callback_handler()]

        self._models["gemini-fast"] = model
        return model

    def _init_claude(self):
        """Initialize and return a Claude model instance."""
        # Check if already initialized and recently used (within last 30 minutes)
        if "claude" in self._models:
            return self._models["claude"]

        # Create and cache the model
        model = ChatAnthropic(
            model_name=settings.CLAUDE_MODEL_NAME,
            temperature=settings.CLAUDE_TEMPERATURE,
            anthropic_api_key=settings.API_KEY_ANTHROPIC,
            max_tokens=settings.CLAUDE_MAX_TOKENS,
        )
        # Attach Langfuse callback handler
        if langfuse_client.get_callback_handler():
            model.callbacks = [langfuse_client.get_callback_handler()] #

        self._models["claude"] = model

        return model

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
        if model_name not in self._model_initializers:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self._model_initializers.keys())}")

        # Return cached model or initialize new one
        return self._model_initializers[model_name]()