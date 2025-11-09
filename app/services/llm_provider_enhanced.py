"""
Enhanced LLM Provider with Per-Call Parameter Control

Allows fine-grained control of temperature, top_p, top_k per invocation
to prevent hallucinations in query generation.
"""

from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings
from pathlib import Path
import os
from google.cloud import aiplatform
import logging
from typing import Optional, Dict, Any
from app.core.langfuse_client import langfuse_client

logger = logging.getLogger(__name__)


class LLMStage:
    """Predefined temperature profiles for different pipeline stages"""

    INTENT_CLASSIFICATION = {
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 10,
        "description": "Deterministic intent classification"
    }

    INTELLIGENT_ANALYSIS = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "description": "Focused analysis with slight creativity"
    }

    QUERY_GENERATION = {
        "temperature": 0.0,
        "top_p": 0.85,
        "top_k": 5,
        "description": "Precise, deterministic query generation"
    }

    QUERY_VALIDATION = {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 10,
        "description": "Thorough but strict validation"
    }

    QUERY_REFINEMENT = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "description": "Creative problem-solving for fixes"
    }

    RETRY_WITH_FEEDBACK = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 30,
        "description": "Explore alternative approaches"
    }

    MEMORY_EXTRACTION = {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 15,
        "description": "Structured extraction from feedback"
    }


class EnhancedLLMProvider:
    """
    Enhanced LLM Provider with per-call parameter control.

    Features:
    - Per-invocation temperature override
    - Stage-specific presets
    - Model parameter adjustment
    - Retry with temperature escalation
    """

    _instance = None
    _models = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnhancedLLMProvider, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._model_initializers = {
                "gemini": self._init_gemini,
                "claude": self._init_claude
            }
            self._initialized = True

    def _init_gemini(self, **kwargs):
        """
        Initialize Gemini model with optional parameter overrides.

        Args:
            **kwargs: Parameter overrides (temperature, top_p, top_k, max_tokens)
        """
        # Get default values from settings
        temperature = kwargs.get("temperature", settings.GEMINI_TEMPERATURE)
        top_p = kwargs.get("top_p", settings.GEMINI_TOP_P)
        top_k = kwargs.get("top_k", 40)  # Default for Gemini
        max_tokens = kwargs.get("max_tokens", 8192)

        # Set up credentials
        credentials_path = settings.GOOGLE_CREDENTIALS_PATH
        credentials_file = Path(credentials_path)
        if not credentials_file.exists():
            raise FileNotFoundError(f"Google credentials file not found at {credentials_path}")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)

        aiplatform.init(
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION
        )

        # Create model with specified parameters
        model = ChatVertexAI(
            model_name=settings.GEMINI_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
            verbose=settings.DEBUG,
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
        )

        # Attach Langfuse callback
        if langfuse_client.get_callback_handler():
            model.callbacks = [langfuse_client.get_callback_handler()]

        logger.info(
            f"Initialized Gemini: temp={temperature}, top_p={top_p}, top_k={top_k}"
        )

        return model

    def _init_claude(self, **kwargs):
        """
        Initialize Claude model with optional parameter overrides.

        Args:
            **kwargs: Parameter overrides (temperature, top_p, max_tokens)
        """
        # Get default values from settings
        temperature = kwargs.get("temperature", settings.CLAUDE_TEMPERATURE)
        top_p = kwargs.get("top_p", 0.95)
        max_tokens = kwargs.get("max_tokens", settings.CLAUDE_MAX_TOKENS)

        # Note: Claude doesn't support top_k, only temperature and top_p

        model = ChatAnthropic(
            model_name=settings.CLAUDE_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            anthropic_api_key=settings.API_KEY_ANTHROPIC,
            max_tokens=max_tokens,
        )

        # Attach Langfuse callback
        if langfuse_client.get_callback_handler():
            model.callbacks = [langfuse_client.get_callback_handler()]

        logger.info(
            f"Initialized Claude: temp={temperature}, top_p={top_p}"
        )

        return model

    def get_model(
        self,
        model_name: str = "gemini",
        stage: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Get a language model with optional parameter overrides.

        Args:
            model_name: Name of the model ("gemini" or "claude")
            stage: Predefined stage (use LLMStage constants)
            temperature: Custom temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (Gemini only)
            max_tokens: Maximum output tokens

        Returns:
            A LangChain chat model instance

        Example:
            # Use stage preset
            model = provider.get_model("gemini", stage=LLMStage.QUERY_GENERATION)

            # Or custom parameters
            model = provider.get_model("gemini", temperature=0.0, top_p=0.85)
        """
        if model_name not in self._model_initializers:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported: {list(self._model_initializers.keys())}"
            )

        # Build parameters dict
        params = {}

        # If stage preset is provided, use its parameters as base
        if stage:
            if hasattr(LLMStage, stage):
                preset = getattr(LLMStage, stage)
                params.update({
                    k: v for k, v in preset.items()
                    if k in ["temperature", "top_p", "top_k", "max_tokens"]
                })

                logger.debug(f"Using {stage} preset: {preset.get('description')}")

        # Override with any explicitly provided parameters
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if top_k is not None:
            params["top_k"] = top_k
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Create cache key including parameters for proper caching
        cache_key = f"{model_name}_{params.get('temperature', 'default')}"

        # Check cache
        if cache_key in self._models:
            cached_model = self._models[cache_key]
            logger.debug(f"Using cached model: {cache_key}")
            return cached_model

        # Initialize new model with parameters
        model = self._model_initializers[model_name](**params)
        self._models[cache_key] = model

        return model

    def get_for_query_generation(self, model_name: str = "gemini"):
        """
        Get model optimized for query generation.
        Uses minimal temperature to prevent hallucinations.
        """
        return self.get_model(
            model_name=model_name,
            stage="QUERY_GENERATION"
        )

    def get_for_validation(self, model_name: str = "gemini"):
        """Get model optimized for query validation."""
        return self.get_model(
            model_name=model_name,
            stage="QUERY_VALIDATION"
        )

    def get_for_refinement(self, model_name: str = "gemini"):
        """Get model optimized for query refinement."""
        return self.get_model(
            model_name=model_name,
            stage="QUERY_REFINEMENT"
        )

    def get_for_retry(self, model_name: str = "gemini"):
        """Get model optimized for retry with feedback."""
        return self.get_model(
            model_name=model_name,
            stage="RETRY_WITH_FEEDBACK"
        )

    async def invoke_with_retry_escalation(
        self,
        model_name: str,
        prompt: str,
        initial_temperature: float = 0.0,
        max_retries: int = 3,
        temperature_increment: float = 0.15
    ) -> str:
        """
        Invoke LLM with automatic temperature escalation on retry.

        Starts with low temperature for precision, gradually increases
        if output is unsatisfactory.

        Args:
            model_name: Model to use
            prompt: Prompt to send
            initial_temperature: Starting temperature
            max_retries: Maximum retry attempts
            temperature_increment: How much to increase temp per retry

        Returns:
            LLM response content
        """
        current_temp = initial_temperature

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries} "
                    f"with temperature={current_temp:.2f}"
                )

                # Get model with current temperature
                model = self.get_model(
                    model_name=model_name,
                    temperature=current_temp,
                    top_p=0.85 if current_temp < 0.3 else 0.95
                )

                # Invoke
                response = await model.ainvoke(prompt)
                content = response.content.strip()

                # Basic validation
                if content and len(content) > 10 and "error" not in content.lower():
                    logger.info(f"Success on attempt {attempt + 1}")
                    return content

                # Escalate temperature for next attempt
                current_temp = min(current_temp + temperature_increment, 0.8)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")

                if attempt == max_retries - 1:
                    raise

                current_temp = min(current_temp + temperature_increment, 0.8)

        raise RuntimeError(f"Failed after {max_retries} attempts")


# Create singleton instance
llm_provider = EnhancedLLMProvider()


# Backward compatibility: provide same interface as original
class LLMProvider(EnhancedLLMProvider):
    """Backward compatible wrapper"""
    pass