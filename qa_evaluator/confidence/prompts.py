"""RAGAS faithfulness 两阶段 prompt。

【阶段 1：从答案抽原子事实】
- 输入：question + answer
- 输出 JSON：{"statements": ["事实1", "事实2", ...]}
- 要求：每条事实是单一可判定的陈述，不含「来源/温馨提示/重要提醒」等元信息

【阶段 2：判定每条事实是否被 retrieved_chunks 支持】
- 输入：单条事实 statement + 全部 retrieved_chunks 拼接
- 输出 JSON：{"verdict": "yes"|"partial"|"no", "reason": "...", "evidence": "..."}
  - yes：事实可由 chunks 直接支持
  - partial：部分支持（关键限定/数字/范围对不上）
  - no：chunks 中找不到依据，或与 chunks 矛盾

faithfulness 分数 = (yes_count + 0.5 * partial_count) / total_statements
高/中/低：≥0.9 / 0.7-0.9 / <0.7
"""
from __future__ import annotations

EXTRACT_SYSTEM = """你是一名 RAG 评测助手。你的任务是把一段「保险问答」的回答拆解成若干条原子事实（statement）。

要求：
1. 每条 statement 是单一、可独立判断真伪的陈述句。
2. 忽略以下元信息，不要拆出来：
   - 「来源：xxx」「【来源：…】」等引用标记
   - 「温馨提示」「重要提醒」「请咨询官方」等模板话术
   - 「现有文档未披露」「以上为现有文档披露的部分内容」等元描述
   - 章节标题、序号编号本身（编号下面的内容才是事实）
3. 拆出的 statement 用简体中文。
4. 信息密集的句子要拆成多条（例如「A 类地区包含澳洲、比利时、加拿大」拆成 3 条）。
5. 只拆「实质性事实」，不要把回答的结构性句子（如「具体要求如下」「关键信息点如下」）拆出来。

输出严格的 JSON：{"statements": ["事实1", "事实2", ...]}。
不要加任何解释。"""


EXTRACT_USER_TEMPLATE = """问题：{question}

回答：
{answer}

请从上述回答中抽出原子事实，按要求输出 JSON。"""


VERIFY_SYSTEM = """你是一名 RAG 评测助手。给你一条「待判定的事实」和若干「检索到的文档片段」，判定该事实是否能由这些片段支撑。

判定标准：
- "yes"：事实可由片段直接支持（关键名词、数字、范围、限定条件全部对得上）
- "partial"：方向对但有偏差（例如片段说「首 10 年」事实说「首 5 年」；或片段只覆盖事实的一部分）
- "no"：片段中找不到依据，或事实与片段矛盾，或事实是 RAG 自行扩展、概括、推理出的内容

注意：
- 评判依据**只能**是给定的检索片段，不能用你自己的常识。
- 即便事实在现实中是真的，只要片段没说，也判 "no"。
- 简体/繁体差异不影响判定（例如「核保」=「核保」）。
- 「来源」标注本身不算事实，已在抽取阶段过滤。

输出严格的 JSON：{"verdict": "yes"|"partial"|"no", "reason": "一句话说明", "evidence": "支撑/反驳片段的最短引用，no 时填空字符串"}。
不要加任何解释。"""


VERIFY_USER_TEMPLATE = """待判定事实：
{statement}

检索到的文档片段：
{chunks_text}

请判定，并按要求输出 JSON。"""


def render_extract_messages(question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_USER_TEMPLATE.format(question=question, answer=answer)},
    ]


def render_verify_messages(statement: str, chunks_text: str) -> list[dict]:
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_USER_TEMPLATE.format(statement=statement, chunks_text=chunks_text)},
    ]


def format_chunks_for_verify(chunks: list[dict]) -> str:
    """把 retrieved_chunks list 拼成一段供 verify 阶段使用的文本。
    保留 rank/source/page/section 用于评判，避免事实被「扩散到不相关 chunk」误判通过。"""
    parts: list[str] = []
    for c in chunks:
        header = (
            f"[片段{c.get('rank')}] 文件={c.get('source','')} 页码={c.get('page','')} "
            f"章节={c.get('section','')}"
        )
        parts.append(header + "\n" + (c.get("content") or ""))
    return "\n\n".join(parts)
