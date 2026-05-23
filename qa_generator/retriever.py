"""从 Chroma 检索目标 chunk 的语义邻居。

batch 模式：传 source_filter（target chunk 的 source 字符串），where 子句精确匹配，
邻居只来自同一文件，避免跨文件召回污染。
v3 单文件模式：source_filter=None，行为与原版一致。
"""
import logging

from config import CHROMA_COLLECTION, EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K
from qa_generator.embedder import _truncate_for_embedding, get_collection
from qa_generator.llm_client import embed

logger = logging.getLogger(__name__)


def query_neighbors(
    target: dict,
    k: int = TOP_K,
    collection_name: str = CHROMA_COLLECTION,
    source_filter: str | None = None,
) -> list[dict]:
    """返回 top-k 语义最相似的其他 chunks（已剔除自身）。

    source_filter: 仅返回 metadata["source"] 精确匹配的邻居（不做 normalize）。

    每个邻居 dict 含: chunk_id, content, content_with_context, source, breadcrumb, has_table, distance
    """
    collection = get_collection(collection_name)
    query_text = _truncate_for_embedding(
        target["content_with_context"], target.get("est_token_count")
    )
    [query_vec] = embed([query_text], model=EMBEDDING_MODEL, dim=EMBEDDING_DIM)

    where = {"source": source_filter} if source_filter else None
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k + 1,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    target_chunk_id = target["chunk_id"]
    neighbors: list[dict] = []
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        # batch 模式 doc_id="source::chunk_id"，从 metadata 拿真实 chunk_id；
        # v3 旧库 metadata 没写 chunk_id 字段，回退用 doc_id 本身。
        cid = (meta or {}).get("chunk_id") or doc_id
        if cid == target_chunk_id:
            continue
        neighbors.append(
            {
                "chunk_id": cid,
                "content": meta.get("content", ""),
                "content_with_context": doc,
                "source": meta.get("source", ""),
                "breadcrumb": meta.get("breadcrumb", ""),
                "has_table": meta.get("has_table", False),
                "distance": dist,
            }
        )
        if len(neighbors) >= k:
            break
    logger.debug(
        "target=%s neighbors=%s",
        target["chunk_id"],
        [(n["chunk_id"], round(n["distance"], 3)) for n in neighbors],
    )
    return neighbors
