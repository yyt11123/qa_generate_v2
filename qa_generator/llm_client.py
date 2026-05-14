"""阿里云百炼 OpenAI 兼容 client 单例。LLM 和 embedding 共用同一个 client。"""
import json
import logging
import os
import random
import time
from functools import lru_cache

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config import (
    DASHSCOPE_BASE_URL,
    LLM_BACKOFF_BASE,
    LLM_MODEL,
    LLM_RETRY_MAX,
    LLM_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL, timeout=LLM_TIMEOUT_SEC)


_RETRYABLE = (APITimeoutError, RateLimitError, APIError, ConnectionError)


def _sleep_backoff(attempt: int) -> None:
    delay = LLM_BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
    time.sleep(delay)


def chat_json(messages: list[dict], model: str = LLM_MODEL, extra_user_hint: str | None = None) -> dict:
    """调 chat completion 强制 JSON 输出。网络/限流错误指数退避重试。

    extra_user_hint: 若 JSON 解析失败重试时，追加到最后一条 user 消息后。
    """
    client = get_client()
    msgs = list(messages)
    if extra_user_hint:
        msgs.append({"role": "user", "content": extra_user_hint})

    last_err: Exception | None = None
    for attempt in range(LLM_RETRY_MAX):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content")
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning("LLM returned non-JSON (attempt %d/%d): %s", attempt + 1, LLM_RETRY_MAX, e)
            raise
        except _RETRYABLE as e:
            last_err = e
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt + 1, LLM_RETRY_MAX, e,
            )
            if attempt + 1 < LLM_RETRY_MAX:
                _sleep_backoff(attempt)
            continue
    assert last_err is not None
    raise last_err


def embed(texts: list[str], model: str, dim: int) -> list[list[float]]:
    """调 embedding API。调用方需自行控制 batch_size <= 10。"""
    client = get_client()
    last_err: Exception | None = None
    for attempt in range(LLM_RETRY_MAX):
        try:
            resp = client.embeddings.create(
                model=model,
                input=texts,
                dimensions=dim,
                encoding_format="float",
            )
            return [d.embedding for d in resp.data]
        except _RETRYABLE as e:
            last_err = e
            logger.warning(
                "Embedding call failed (attempt %d/%d): %s",
                attempt + 1, LLM_RETRY_MAX, e,
            )
            if attempt + 1 < LLM_RETRY_MAX:
                _sleep_backoff(attempt)
            continue
    assert last_err is not None
    raise last_err
