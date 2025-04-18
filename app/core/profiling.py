# app/core/profiling.py
import asyncio
import time
import functools
from app.core.logging import logger

def timeit(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ {func.__module__}.{func.__name__} took {elapsed_time:.4f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ {func.__module__}.{func.__name__} took {elapsed_time:.4f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper