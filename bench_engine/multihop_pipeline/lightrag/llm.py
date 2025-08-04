import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Dict, Any, Callable, List
import openai
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("LLMService")

handler = TimedRotatingFileHandler(
    "llm_service.log", when="midnight", interval=1, backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_api_call(func_name: str, metadata: Dict[str, Any], is_error: bool = False):
    log_level = logging.ERROR if is_error else logging.INFO
    logger.log(log_level, f"{func_name} called with metadata: {metadata}")


async def gpt_4o_mini_complete(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = openai.AsyncOpenAI()
    hashing_kv = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = hash(str(messages))
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, **kwargs
    )

    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": "gpt-4o-mini"}})
    return result

async def gpt_4o_complete(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = openai.AsyncOpenAI()
    hashing_kv = kwargs.pop("hashing_kv", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = hash(str(messages))
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model="gpt-4o", messages=messages, **kwargs
    )

    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": "gpt-4o-mini"}})
    return result


class Model(BaseModel):
    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True