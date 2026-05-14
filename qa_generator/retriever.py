"""从 Chroma 检索目标 chunk 的语义邻居。"""
import logging

from config import EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K
from qa_generator.embedder import _truncate_for_embedding, get_collection
from qa_generator.llm_client import embed

logger = logging.getLogger(__name__)


def query_neighbors(target: dict, k: int = TOP_K) -> list[dict]:
    """返回 top-k 语义最相似的其他 chunks（已剔除自身）。

    每个邻居 dict 含: chunk_id, content, content_with_context, source, breadcrumb, has_table, distance
    """
    collection = get_collection()
    query_text = _truncate_for_embedding(
        target["content_with_context"], target.get("est_token_count")
    )
    [query_vec] = embed([query_text], model=EMBEDDING_MODEL, dim=EMBEDDING_DIM)

    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k + 1,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    neighbors: list[dict] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        if cid == target["chunk_id"]:
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
