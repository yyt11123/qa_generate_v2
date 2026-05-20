"""9 个机器指标。所有指标无 API 调用，纯本地计算。
9 个指标分别是：答案字符在支撑信息里的占比 ≥ 50%，支撑信息真的来自原 chunk ≥ 95%，
不能混入简体字（繁体业务），不能出现"本节/上述"等元问题 ≤ 5%，答案开头不能用模糊指代 ≤ 10%，
8 大业务分类至少覆盖 5 类，最大类占比 ≤ 40%，防一边倒，至少 85% 的 chunk 被出过题，
同 chunk 内重复率 ≤ 5%
阈值常量集中在 THRESHOLDS 中；用户审阅后可调整。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ===== 阈值（聚合层，决定 metric.passed）=====
THRESHOLDS: dict[str, dict[str, Any]] = {
    "answer_groundedness": {"op": ">=", "value": 0.50, "severity": "warning"},
    "sf_in_chunk_rate": {"op": ">=", "value": 0.95, "severity": "failure"},
    "simplified_pollution_rate": {"op": "<=", "value": 0.0, "severity": "failure"},
    "meta_question_rate": {"op": "<=", "value": 0.05, "severity": "failure"},
    "vague_reference_rate": {"op": "<=", "value": 0.10, "severity": "warning"},
    "category_coverage": {"op": ">=", "value": 0.625, "severity": "warning"},  # ≥5/8
    "category_balance": {"op": "<=", "value": 0.40, "severity": "warning"},
    "chunk_coverage": {"op": ">=", "value": 0.85, "severity": "failure"},
    "duplicate_qa_rate": {"op": "<=", "value": 0.05, "severity": "warning"},
}

# ===== 单条 QA 违规阈值（决定违规明细中 included / not）=====
PER_ITEM = {
    "answer_groundedness": 0.50,  # < 此值 → 列入明细
    "sf_in_chunk_rate": 0.80,  # 即用户给的 ≥0.8 视为来自 chunk
    "duplicate_qa_pair": 0.70,  # 同 chunk 内 question 相似度 > 此值 → 重复
}

# ===== 关键词 / 字符集 =====
SIMPLIFIED_CHARSET = set("为产业规则财务缴费应该这国时间发现实际见证书审过专门")
META_KEYWORDS = ["本节", "上述", "以下", "此章", "本章"]
VAGUE_REFERENCES = ["這種", "該項", "此種", "上述"]


def _passes(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(f"unknown op: {op}")


def _wrap(name: str, value: float, violations: list[dict]) -> dict:
    cfg = THRESHOLDS[name]
    return {
        "name": name,
        "value": round(value, 4),
        "threshold": cfg["value"],
        "op": cfg["op"],
        "severity": cfg["severity"],
        "passed": _passes(value, cfg["op"], cfg["value"]),
        "violations": violations,
    }


# ============================================================
# A 类：数据可信度
# ============================================================
def metric_answer_groundedness(rows: list[dict]) -> dict:
    """answer 中字符出现在 supporting_facts 的比例（去除空白）。
    每条 QA 算一个比值，最终输出全集平均值。"""
    per_qa_ratios = []
    violations = []
    for r in rows:
        a = (r.get("answer") or "").replace(" ", "").replace("\n", "").replace("\t", "")
        sf = (
            (r.get("supporting_facts") or "")
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
        )
        if not a:
            ratio = 0.0
        else:
            sf_chars = set(sf)
            ratio = sum(1 for c in a if c in sf_chars) / len(a)
        per_qa_ratios.append(ratio)
        if ratio < PER_ITEM["answer_groundedness"]:
            violations.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "question": (r.get("question") or "")[:80],
                    "ratio": round(ratio, 3),
                }
            )
    avg = sum(per_qa_ratios) / len(per_qa_ratios) if per_qa_ratios else 0.0
    return _wrap("answer_groundedness", avg, violations)


def _sf_containment(sf: str, content: str) -> float:
    """返回 sf 字符匹配进 content 的比例，用 SequenceMatcher 求 matching blocks 总长 / len(sf)。
    长 vs 长直接 .ratio() 在长度差大时偏低，所以这里用 matching block sum 更贴近"是否包含"语义。"""
    sf = (sf or "").replace(" ", "").replace("\n", "")
    content = (content or "").replace(" ", "").replace("\n", "")
    if not sf:
        return 0.0
    sm = SequenceMatcher(None, sf, content, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / len(sf)


def metric_sf_in_chunk_rate(rows: list[dict], chunks_by_id: dict[str, dict]) -> dict:
    """对每条 QA 求 supporting_facts 与所属 chunk content 的 SequenceMatcher 相似度。
    相似度 ≥0.8 视为"来自 chunk"。输出比例（来自 chunk 的 QA / 总数）。"""
    in_count = 0
    violations = []
    for r in rows:
        cid = r.get("chunk_id")
        chunk = chunks_by_id.get(cid)
        content = (chunk or {}).get("content", "")
        sim = _sf_containment(r.get("supporting_facts") or "", content)
        if sim >= PER_ITEM["sf_in_chunk_rate"]:
            in_count += 1
        else:
            violations.append(
                {
                    "chunk_id": cid,
                    "question": (r.get("question") or "")[:80],
                    "similarity": round(sim, 3),
                }
            )
    rate = in_count / len(rows) if rows else 0.0
    return _wrap("sf_in_chunk_rate", rate, violations)


def metric_simplified_pollution_rate(rows: list[dict]) -> dict:
    """扫 question + answer 中是否含特定简体字符集。"""
    polluted = []
    for r in rows:
        text = (r.get("question") or "") + (r.get("answer") or "")
        hits = sorted(set(c for c in text if c in SIMPLIFIED_CHARSET))
        if hits:
            polluted.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "question": (r.get("question") or "")[:80],
                    "simplified_chars": hits,
                }
            )
    rate = len(polluted) / len(rows) if rows else 0.0
    return _wrap("simplified_pollution_rate", rate, polluted)


def metric_meta_question_rate(rows: list[dict]) -> dict:
    """question 中含元问题关键词的比例。"""
    flagged = []
    for r in rows:
        q = r.get("question") or ""
        hits = [k for k in META_KEYWORDS if k in q]
        if hits:
            flagged.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "question": q[:120],
                    "hit_keywords": hits,
                }
            )
    rate = len(flagged) / len(rows) if rows else 0.0
    return _wrap("meta_question_rate", rate, flagged)


def metric_vague_reference_rate(rows: list[dict]) -> dict:
    """answer 开头 20 字内含模糊指代关键词的比例（warning 级别，会有误伤）。"""
    flagged = []
    for r in rows:
        a = (r.get("answer") or "")[:20]
        hits = [k for k in VAGUE_REFERENCES if k in a]
        if hits:
            flagged.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "answer_head": a,
                    "hit_keywords": hits,
                }
            )
    rate = len(flagged) / len(rows) if rows else 0.0
    return _wrap("vague_reference_rate", rate, flagged)


# ============================================================
# C 类：覆盖均衡性
# ============================================================
def metric_category_coverage(rows: list[dict], total_categories: int = 8) -> dict:
    """覆盖的分类数 / 总分类数（8）。xlsx 中分类已转简体。"""
    cats = set(r.get("category_simp") or r.get("category") or "" for r in rows)
    cats.discard("")
    coverage = len(cats) / total_categories
    violations = []
    all_simp = [
        "案例",
        "产品",
        "投保规则",
        "健康核保",
        "财务核保",
        "缴费",
        "行政规则",
        "一般查询",
    ]
    missing = [c for c in all_simp if c not in cats]
    if missing:
        violations.append({"missing_categories": missing})
    return _wrap("category_coverage", coverage, violations)


def metric_category_balance(rows: list[dict]) -> dict:
    """最大分类条数 / 总条数。"""
    counter = Counter((r.get("category_simp") or r.get("category") or "") for r in rows)
    if not counter:
        return _wrap("category_balance", 0.0, [])
    most_cat, most_n = counter.most_common(1)[0]
    ratio = most_n / sum(counter.values())
    violations = []
    if not _passes(ratio, "<=", THRESHOLDS["category_balance"]["value"]):
        violations.append(
            {"dominant_category": most_cat, "count": most_n, "ratio": round(ratio, 3)}
        )
    return _wrap("category_balance", ratio, violations)


def metric_chunk_coverage(rows: list[dict], total_chunks: int) -> dict:
    """有 QA 的 chunk_id 唯一数 / 总 chunk 数。"""
    seen = set(r.get("chunk_id") for r in rows if r.get("chunk_id"))
    coverage = len(seen) / total_chunks if total_chunks else 0.0
    return _wrap("chunk_coverage", coverage, [])


def metric_duplicate_qa_rate(rows: list[dict]) -> dict:
    """同 chunk_id 内 question 相似度 > 0.7 的对数 × 2 / 总条数。"""
    by_chunk: dict[str, list[dict]] = {}
    for r in rows:
        by_chunk.setdefault(r.get("chunk_id") or "", []).append(r)
    dup_pairs = 0
    violations = []
    for cid, group in by_chunk.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                q1 = group[i].get("question") or ""
                q2 = group[j].get("question") or ""
                sim = SequenceMatcher(None, q1, q2, autojunk=False).ratio()
                if sim > PER_ITEM["duplicate_qa_pair"]:
                    dup_pairs += 1
                    violations.append(
                        {
                            "chunk_id": cid,
                            "q1": q1[:80],
                            "q2": q2[:80],
                            "similarity": round(sim, 3),
                        }
                    )
    rate = dup_pairs * 2 / len(rows) if rows else 0.0
    return _wrap("duplicate_qa_rate", rate, violations)


# ============================================================
# 总入口
# ============================================================
def run_all(rows: list[dict], chunks_by_id: dict[str, dict], total_chunks: int) -> dict:
    """跑 9 个指标，返回 dict 结构 {metric_name: metric_dict}."""
    results = {
        "answer_groundedness": metric_answer_groundedness(rows),
        "sf_in_chunk_rate": metric_sf_in_chunk_rate(rows, chunks_by_id),
        "simplified_pollution_rate": metric_simplified_pollution_rate(rows),
        "meta_question_rate": metric_meta_question_rate(rows),
        "vague_reference_rate": metric_vague_reference_rate(rows),
        "category_coverage": metric_category_coverage(rows),
        "category_balance": metric_category_balance(rows),
        "chunk_coverage": metric_chunk_coverage(rows, total_chunks),
        "duplicate_qa_rate": metric_duplicate_qa_rate(rows),
    }
    return results
