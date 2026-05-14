"""Embedding + Chroma 持久化封装。按 chunk_id 幂等。"""
import logging
from typing import Iterable

import chromadb

from config import (
    CHROMA_COLLECTION,
    EMBEDDING_BATCH,
    EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
)
from qa_generator.llm_client import embed

logger = logging.getLogger(__name__)


def _truncate_for_embedding(text: str, est_tokens: int | None) -> str:
    """text-embedding-v4 单文本上限 8192 tokens。est_token_count 是上游粗估，留点余量。"""
    if est_tokens is not None and est_tokens <= EMBEDDING_MAX_TOKENS:
        return text
    char_budget = EMBEDDING_MAX_TOKENS * 3
    if len(text) > char_budget:
        logger.warning(
            "text too long (est_tokens=%s, chars=%d), truncating to %d chars",
            est_tokens, len(text), char_budget,
        )
        return text[:char_budget]
    return text


def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _existing_ids(collection, ids: Iterable[str]) -> set[str]:
    ids = list(ids)
    if not ids:
        return set()
    got = collection.get(ids=ids, include=[])
    return set(got["ids"])


def _batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def index_chunks(chunks: list[dict]) -> dict:
    """为 chunks 建索引。已存在的 chunk_id 跳过，不重复调 embedding API。

    返回统计 dict: {"total": N, "embedded": M, "skipped": K}.
    """
    collection = get_collection()
    all_ids = [c["chunk_id"] for c in chunks]
    already = _existing_ids(collection, all_ids)
    to_embed = [c for c in chunks if c["chunk_id"] not in already]

    logger.info(
        "vector index: total=%d already_indexed=%d to_embed=%d",
        len(chunks), len(already), len(to_embed),
    )

    embedded_count = 0
    for batch in _batched(to_embed, EMBEDDING_BATCH):
        texts = [
            _truncate_for_embedding(c["content_with_context"], c.get("est_token_count"))
            for c in batch
        ]
        vectors = embed(texts, model=EMBEDDING_MODEL, dim=EMBEDDING_DIM)
        ids = [c["chunk_id"] for c in batch]
        documents = [c["content_with_context"] for c in batch]
        metadatas = [
            {
                "source": c.get("source", ""),
                "section_title": c.get("section_title", "") or "",
                "breadcrumb": " > ".join(c.get("breadcrumb") or []),
                "has_table": bool(c.get("has_table", False)),
                "content": c.get("content", ""),
            }
            for c in batch
        ]
        collection.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
        embedded_count += len(batch)
        logger.info("embedded batch of %d (running total %d)", len(batch), embedded_count)

    return {"total": len(chunks), "embedded": embedded_count, "skipped": len(already)}
