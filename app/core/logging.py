# app/core/logging.py
import logging
import sys
from app.core.config import settings

# Configure logger
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("query-generator")

# Set higher logging levels for verbose libraries
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
# You might also consider adding "langfuse" if it has its own verbose logger
logging.getLogger("langfuse").setLevel(logging.INFO)
