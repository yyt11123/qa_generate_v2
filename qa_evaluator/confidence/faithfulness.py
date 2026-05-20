"""RAGAS faithfulness 主逻辑：抽事实 + 逐条判定 + 聚合分数。

【流程】
对每条解析后的 QA：
  1. extract_statements(question, answer) → list[statement]
     调用 1 次 LLM。
  2. 对每条 statement：verify_statement(statement, chunks)
     调用 N 次 LLM（N = statement 数）。
  3. 聚合：score = (yes + 0.5 * partial) / total
     等级：≥0.9 high / 0.7-0.9 medium / <0.7 low

【LLM 调用统计】
total_calls = 1 + N（每条 QA）
4 条 QA 估算：4 * (1 + 平均 statement 数) ≈ 30-50 次

判定模型走 config.LLM_MODEL（qwen-plus）。
"""
from __future__ import annotations

import logging
from typing import Any

from qa_generator.llm_client import chat_json
from qa_evaluator.confidence.prompts import (
    format_chunks_for_verify,
    render_extract_messages,
    render_verify_messages,
)

logger = logging.getLogger(__name__)

VERDICT_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def classify_score(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.7:
        return "medium"
    return "low"


def extract_statements(question: str, answer: str) -> list[str]:
    """阶段 1：从答案里抽原子事实。返回字符串列表。"""
    msgs = render_extract_messages(question, answer)
    data = chat_json(msgs)
    raw = data.get("statements") or []
    statements = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    if not statements:
        logger.warning("extract_statements got 0 statements, raw=%s", data)
    return statements


def verify_statement(statement: str, chunks: list[dict]) -> dict:
    """阶段 2：对单条 statement 判 yes/partial/no。
    返回 {"verdict": str, "reason": str, "evidence": str}。"""
    chunks_text = format_chunks_for_verify(chunks)
    msgs = render_verify_messages(statement, chunks_text)
    data = chat_json(msgs)
    verdict = (data.get("verdict") or "").strip().lower()
    if verdict not in VERDICT_SCORE:
        logger.warning("unexpected verdict %r, treating as 'no'", verdict)
        verdict = "no"
    return {
        "verdict": verdict,
        "reason": (data.get("reason") or "").strip(),
        "evidence": (data.get("evidence") or "").strip(),
    }


def evaluate_one(qa: dict) -> dict:
    """跑单条 QA 的 faithfulness 评估。返回详细结果 dict（含每条 statement 判定）。"""
    qid = qa.get("question_id")
    question = qa.get("question") or ""
    answer = qa.get("answer") or ""
    chunks = qa.get("retrieved_chunks") or []

    statements = extract_statements(question, answer)
    n_total = len(statements)
    if n_total == 0:
        return {
            "question_id": qid,
            "question": question,
            "n_statements": 0,
            "n_yes": 0,
            "n_partial": 0,
            "n_no": 0,
            "score": 0.0,
            "level": "low",
            "statements": [],
            "llm_calls": 1,
        }

    per_statement: list[dict] = []
    counts = {"yes": 0, "partial": 0, "no": 0}
    for stmt in statements:
        v = verify_statement(stmt, chunks)
        counts[v["verdict"]] += 1
        per_statement.append({"statement": stmt, **v})

    score = (counts["yes"] + 0.5 * counts["partial"]) / n_total
    return {
        "question_id": qid,
        "question": question,
        "n_statements": n_total,
        "n_yes": counts["yes"],
        "n_partial": counts["partial"],
        "n_no": counts["no"],
        "score": round(score, 4),
        "level": classify_score(score),
        "statements": per_statement,
        "llm_calls": 1 + n_total,
    }


def evaluate_all(parsed: list[dict]) -> dict[str, Any]:
    """跑全部 QA，返回 {"per_question": [...], "summary": {...}}。"""
    per_q: list[dict] = []
    total_calls = 0
    for qa in parsed:
        res = evaluate_one(qa)
        total_calls += res["llm_calls"]
        per_q.append(res)
        logger.info(
            "Q%s done: score=%s level=%s (yes=%d partial=%d no=%d)",
            res["question_id"], res["score"], res["level"],
            res["n_yes"], res["n_partial"], res["n_no"],
        )

    if per_q:
        avg = sum(r["score"] for r in per_q) / len(per_q)
    else:
        avg = 0.0
    summary = {
        "n_questions": len(per_q),
        "avg_score": round(avg, 4),
        "avg_level": classify_score(avg),
        "total_llm_calls": total_calls,
    }
    return {"per_question": per_q, "summary": summary}
