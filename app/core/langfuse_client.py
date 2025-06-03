# app/core/langfuse_client.py
from langfuse import Langfuse #
from langfuse.callback import CallbackHandler #
from app.core.config import settings
from app.core.logging import logger

class LangfuseClient:
    _instance = None
    _langfuse = None
    _callback_handler = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LangfuseClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if settings.LANGFUSE_ACTIVATE and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                self._langfuse = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                self._callback_handler = CallbackHandler(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                logger.info("Langfuse client and callback handler initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self._langfuse = None
                self._callback_handler = None
        else:
            logger.info("Langfuse not activated or credentials missing.")
            self._langfuse = None
            self._callback_handler = None

    def get_client(self) -> Langfuse: #
        return self._langfuse

    def get_callback_handler(self) -> CallbackHandler: #
        return self._callback_handler

# Instantiate the singleton client
langfuse_client = LangfuseClient()