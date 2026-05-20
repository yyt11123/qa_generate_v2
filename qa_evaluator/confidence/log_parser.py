"""解析同学 RAG 系统的批量检索日志 → 结构化 JSON。

【输入格式约定】
inputs/full_retrieval_batch_test_output.txt 是 RAG 系统的运行日志，包含：
- 4 个 QUESTION 块，分隔标记 `QUESTION N:` ... `END QUESTION N`
- 每个块内含若干 RESULT 子块：`--- RESULT N BEGIN ---` ... `--- RESULT N END ---`
  RESULT 内含：`[N] score=... type=...` 一行 + `公司=... | 文件=... | 页码=... | 章节=...` 一行 + `完整原文：` 后跟若干行原文
- 每个块结尾的最终回答：`FINAL ANSWER:` 之后到 `END QUESTION N` 之前

【时间戳】
所有 agent 日志行带 `2026-05-19 HH:MM:SS | INFO     | agent:222 | ` 前缀，需剥离。
FINAL ANSWER 部分是裸文本（无前缀），原样保留。

【输出】
output/rag_logs_parsed.json：
[
  {
    "question_id": 1,
    "question": "...",
    "retrieved_chunks": [
       {"rank": 1, "score": 0.5866, "type": "semantic",
        "company": "安达", "source": "...", "page": "8", "section": "...",
        "content": "..."},
       ...
    ],
    "answer": "..."
  },
  ...
]

无 API 调用。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s*\|\s*\w+\s*\|\s*[\w:]+\s*\|\s?"
)
QUESTION_HEADER_RE = re.compile(r"^QUESTION\s+(\d+):\s*(.+)$")
QUESTION_END_RE = re.compile(r"^END QUESTION\s+(\d+)\s*$")
RESULT_BEGIN_RE = re.compile(r"^---\s*RESULT\s+(\d+)\s+BEGIN\s*---\s*$")
RESULT_END_RE = re.compile(r"^---\s*RESULT\s+(\d+)\s+END\s*---\s*$")
SCORE_LINE_RE = re.compile(r"^\[(\d+)\]\s+score=([\d.]+)\s+type=(\S+)\s*$")
META_LINE_RE = re.compile(
    r"公司=(.*?)\s*\|\s*文件=(.*?)\s*\|\s*页码=(.*?)\s*\|\s*章节=(.*)$"
)
CONTENT_MARK = "完整原文:"


def _strip_log_prefix(line: str) -> str:
    return LOG_PREFIX_RE.sub("", line, count=1)


def _parse_chunk_block(lines: list[str]) -> dict | None:
    """RESULT N BEGIN/END 之间的内容（已剥前缀）。"""
    score_match = None
    meta_match = None
    content_lines: list[str] = []
    in_content = False
    for ln in lines:
        s = ln.strip()
        if not in_content:
            m = SCORE_LINE_RE.match(s)
            if m:
                score_match = m
                continue
            m = META_LINE_RE.search(s)
            if m:
                meta_match = m
                continue
            if s == CONTENT_MARK or s.startswith(CONTENT_MARK):
                in_content = True
                continue
        else:
            content_lines.append(ln)
    if score_match is None or meta_match is None:
        logger.warning("chunk block missing score/meta line, skipping")
        return None
    rank = int(score_match.group(1))
    score = float(score_match.group(2))
    rtype = score_match.group(3)
    company, source, page, section = (g.strip() for g in meta_match.groups())
    content = "\n".join(content_lines).strip()
    # 同学日志里的 chunk 原文是把换行写成字面 "\n"（两个字符），
    # 这里还原为真换行，避免 faithfulness 阶段 LLM 把它当噪声。
    content = content.replace("\\n", "\n")
    return {
        "rank": rank,
        "score": score,
        "type": rtype,
        "company": company,
        "source": source,
        "page": page,
        "section": section,
        "content": content,
    }


def _parse_question_block(qid: int, question: str, body: list[str]) -> dict:
    """body：QUESTION N: 标题之后到 END QUESTION N 之前的所有行（原始未剥前缀）。
    内含若干 RESULT 子块（带 agent 日志前缀）+ 一段 FINAL ANSWER（裸文本）。"""
    chunks: list[dict] = []
    answer_lines: list[str] = []

    i = 0
    n = len(body)
    in_final_answer = False
    while i < n:
        raw = body[i]
        stripped = _strip_log_prefix(raw)
        s = stripped.strip()

        # FINAL ANSWER 是裸文本，不会有 agent 日志前缀
        if not in_final_answer and raw.strip() == "FINAL ANSWER:":
            in_final_answer = True
            i += 1
            continue

        if in_final_answer:
            answer_lines.append(raw.rstrip("\n"))
            i += 1
            continue

        m = RESULT_BEGIN_RE.match(s)
        if m:
            j = i + 1
            inner: list[str] = []
            while j < n:
                inner_stripped = _strip_log_prefix(body[j]).strip()
                if RESULT_END_RE.match(inner_stripped):
                    break
                inner.append(_strip_log_prefix(body[j]).rstrip("\n"))
                j += 1
            chunk = _parse_chunk_block(inner)
            if chunk is not None:
                chunks.append(chunk)
            i = j + 1
            continue

        i += 1

    chunks.sort(key=lambda c: c["rank"])
    answer = "\n".join(answer_lines).strip()
    return {
        "question_id": qid,
        "question": question.strip(),
        "retrieved_chunks": chunks,
        "answer": answer,
    }


def parse_log(log_path: str | Path) -> list[dict]:
    log_path = Path(log_path)
    text = log_path.read_text(encoding="utf-8")
    raw_lines = text.splitlines()

    questions: list[dict] = []
    i = 0
    n = len(raw_lines)
    while i < n:
        line = raw_lines[i].strip()
        m = QUESTION_HEADER_RE.match(line)
        if not m:
            i += 1
            continue
        qid = int(m.group(1))
        qtext = m.group(2)
        body: list[str] = []
        j = i + 1
        while j < n:
            end_m = QUESTION_END_RE.match(raw_lines[j].strip())
            if end_m and int(end_m.group(1)) == qid:
                break
            body.append(raw_lines[j])
            j += 1
        questions.append(_parse_question_block(qid, qtext, body))
        i = j + 1

    questions.sort(key=lambda q: q["question_id"])
    return questions


def write_parsed(parsed: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    src = Path("inputs/full_retrieval_batch_test_output.txt")
    dst = Path("output/rag_logs_parsed.json")
    parsed = parse_log(src)
    write_parsed(parsed, dst)
    print(f"Parsed {len(parsed)} questions → {dst}")
    for q in parsed:
        print(
            f"  Q{q['question_id']}: chunks={len(q['retrieved_chunks'])}, "
            f"answer_chars={len(q['answer'])}"
        )


if __name__ == "__main__":
    main()
