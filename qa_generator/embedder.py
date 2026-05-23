"""Embedding + Chroma 持久化封装。按 doc_id 幂等。

【v3 vs batch 模式】
- v3 单文件：collection=CHROMA_COLLECTION ("chunks_v3")，doc_id 直接用 chunk_id（保持原行为不变）。
- batch 多文件：collection=CHROMA_COLLECTION_BATCH ("chunks_batch")，
  doc_id = source + "::" + chunk_id，避免不同文件 chunk_id 撞车。
  metadata 含 source 字段，供检索时按 source 过滤。

两个 collection 完全独立、各自持久化，v3 已建好的索引不受影响。
"""
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


def get_collection(collection_name: str = CHROMA_COLLECTION):
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_or_create_collection(
        name=collection_name,
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


def _make_doc_id(chunk: dict, scope_by_source: bool) -> str:
    """batch 模式下 doc_id 加 source 前缀避免撞车；v3 模式保持原 chunk_id。"""
    if scope_by_source:
        return f"{chunk.get('source','')}::{chunk['chunk_id']}"
    return chunk["chunk_id"]


def index_chunks(
    chunks: list[dict],
    collection_name: str = CHROMA_COLLECTION,
    scope_by_source: bool = False,
) -> dict:
    """为 chunks 建索引。已存在的 doc_id 跳过，不重复调 embedding API。

    scope_by_source: True 时 doc_id = "<source>::<chunk_id>"（batch 多文件模式用）。

    返回统计 dict: {"total": N, "embedded": M, "skipped": K}.
    """
    collection = get_collection(collection_name)
    all_ids = [_make_doc_id(c, scope_by_source) for c in chunks]
    already = _existing_ids(collection, all_ids)
    pairs = [(c, did) for c, did in zip(chunks, all_ids) if did not in already]

    logger.info(
        "vector index (%s): total=%d already_indexed=%d to_embed=%d",
        collection_name, len(chunks), len(already), len(pairs),
    )

    embedded_count = 0
    for batch in _batched(pairs, EMBEDDING_BATCH):
        chunk_batch = [p[0] for p in batch]
        id_batch = [p[1] for p in batch]
        texts = [
            _truncate_for_embedding(c["content_with_context"], c.get("est_token_count"))
            for c in chunk_batch
        ]
        vectors = embed(texts, model=EMBEDDING_MODEL, dim=EMBEDDING_DIM)
        documents = [c["content_with_context"] for c in chunk_batch]
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "source": c.get("source", "") or "",
                "section_title": c.get("section_title", "") or "",
                "breadcrumb": " > ".join(c.get("breadcrumb") or []),
                "has_table": bool(c.get("has_table", False)),
                "content": c.get("content", ""),
            }
            for c in chunk_batch
        ]
        collection.add(ids=id_batch, embeddings=vectors, documents=documents, metadatas=metadatas)
        embedded_count += len(batch)
        logger.info("embedded batch of %d (running total %d)", len(batch), embedded_count)

    return {"total": len(chunks), "embedded": embedded_count, "skipped": len(already)}
