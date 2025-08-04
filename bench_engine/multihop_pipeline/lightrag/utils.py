import hashlib
import logging
import asyncio
import re
import tiktoken
from typing import Callable, List, Any
from functools import wraps

logger = logging.getLogger("lightrag")


def set_logger(log_file):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode()).hexdigest()


def clean_str(input_str: str) -> str:
    if input_str is None:
        return ""
    return input_str.strip().strip('"').strip("'")


def is_float_regex(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def split_string_by_multi_markers(content: str, markers: List[str]) -> List[str]:
    if not markers:
        return [content]

    pattern = '|'.join(re.escape(marker) for marker in markers)
    parts = re.split(pattern, content)
    return [part.strip() for part in parts if part.strip()]


def encode_string_by_tiktoken(text: str, model_name: str = "gpt-4o") -> List[int]:
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding.encode(text)


def decode_tokens_by_tiktoken(tokens: List[int], model_name: str = "gpt-4o") -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    return encoding.decode(tokens)


def limit_async_func_call(max_async: int):
    def decorator(func: Callable):
        semaphore = asyncio.Semaphore(max_async)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


EmbeddingFunc = Callable[[List[str]], List[List[float]]]