"""调 LLM 生成 QA + 解析校验。"""
import json
import logging

from config import CATEGORIES, LLM_JSON_RETRY, SUPPORTING_FACTS_OVERLAP_MIN, TYPE_VALUES
from qa_generator.llm_client import chat_json
from qa_generator.qa_prompt import build_messages, json_retry_hint

logger = logging.getLogger(__name__)


def _char_overlap_ratio(snippet: str, source: str) -> float:
    """简单的字符级重叠度：snippet 中有多少比例的 4-gram 出现在 source。"""
    snippet = (snippet or "").strip()
    source = (source or "").strip()
    if not snippet:
        return 0.0
    if snippet in source:
        return 1.0
    n = 4
    if len(snippet) < n:
        return 1.0 if snippet in source else 0.0
    grams = {snippet[i : i + n] for i in range(len(snippet) - n + 1)}
    if not grams:
        return 0.0
    hit = sum(1 for g in grams if g in source)
    return hit / len(grams)


def _validate_qa(qa: dict, target_content: str, target_has_table: bool, chunk_id: str) -> dict | None:
    """校验单个 QA。结构非法返回 None；分类/类型不合法尝试纠正；supporting_facts 重叠度低记 WARN 但保留。"""
    required = ("category", "question", "answer", "supporting_facts", "type")
    for k in required:
        if k not in qa or not isinstance(qa[k], str):
            logger.warning("[%s] qa missing/invalid field %s, dropping: %r", chunk_id, k, qa)
            return None

    if qa["category"] not in CATEGORIES:
        logger.warning(
            "[%s] invalid category %r, falling back to '一般查詢'", chunk_id, qa["category"]
        )
        qa["category"] = "一般查詢"

    if qa["type"] not in TYPE_VALUES:
        logger.warning(
            "[%s] invalid type %r, falling back to %s",
            chunk_id, qa["type"], "table" if target_has_table else "text",
        )
        qa["type"] = "table" if target_has_table else "text"

    overlap = _char_overlap_ratio(qa["supporting_facts"], target_content)
    if overlap < SUPPORTING_FACTS_OVERLAP_MIN:
        logger.warning(
            "[%s] supporting_facts low overlap with target chunk: overlap=%.3f, snippet=%r",
            chunk_id, overlap, qa["supporting_facts"][:80],
        )
    qa["_overlap"] = round(overlap, 3)
    return qa


def generate_qa_for_chunk(target: dict, neighbors: list[dict]) -> list[dict]:
    """生成 QA。返回 list[dict]，每个 dict 含 category/question/answer/supporting_facts/type/_overlap。
    返回空列表表示 chunk 信息量不足，跳过。
    """
    chunk_id = target["chunk_id"]
    messages = build_messages(target, neighbors)

    raw: dict | None = None
    last_err: Exception | None = None
    for json_attempt in range(LLM_JSON_RETRY + 1):
        try:
            raw = chat_json(
                messages,
                extra_user_hint=json_retry_hint() if json_attempt > 0 else None,
            )
            break
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning(
                "[%s] LLM returned non-JSON (json_attempt %d/%d)",
                chunk_id, json_attempt + 1, LLM_JSON_RETRY + 1,
            )
            continue
    if raw is None:
        raise last_err if last_err else RuntimeError(f"[{chunk_id}] failed to get JSON")

    qa_list = raw.get("qa_pairs", [])
    if not isinstance(qa_list, list):
        raise ValueError(f"[{chunk_id}] qa_pairs is not a list: {raw!r}")

    validated: list[dict] = []
    for qa in qa_list[:3]:
        if not isinstance(qa, dict):
            logger.warning("[%s] qa item is not a dict, skipping: %r", chunk_id, qa)
            continue
        v = _validate_qa(qa, target["content"], bool(target.get("has_table", False)), chunk_id)
        if v is not None:
            validated.append(v)

    return validated
