"""分层抽样：从 75 条 QA 中抽 20 条供人工标注。

【约束分级】

硬约束（必满足）：
- 总数 = 20
- 跨页 QA ≥ 2 条

软约束（尽量满足，记录偏差）：
- 分类配额（健康核保 4 / 案例 3 / 财务核保 3 / 行政规则 3 / 投保规则 3 / 一般查询 2）
  跨页 QA 落入哪类会让该类 +1~+2，是为保证跨页覆盖的代价
- 每个 chunk_id ≤ 2 次
  受数据本身分类多样性限制：当某分类全集只分布在 1-2 个 chunk 时，
  分类配额 + 跨页配额会让该 chunk 被多次选中
  （v3 例：一般查询全集只有 3 条 QA，分布在 chunk_0024 + chunk_0028 两个 chunk，
   配额 2 条时 chunk_0024 被选 ≥1 次几乎必然，叠加跨页 / 行政规则配额会触达 ≥3 次）

固定 random_state=42，可重现。

实现说明：
- 跨页 2 条优先从所有跨页 QA 中按 random_state 抽。
- 各分类按需求抽，跳过已被跨页选中的（不重复）。
- 若某分类 QA 总数不足配额，从其他分类按现有占比补齐到 20 条。
- 不会重复抽同一条 QA（按 (chunk_id, question) 唯一标识）。
- 抽样结束打印两类 warning：
    1) 分类配额偏差（与跨页 QA 落入的分类有关）
    2) chunk 多样性偏差（某 chunk_id 出现 ≥3 次时）
"""
from __future__ import annotations

import logging
import random
from collections import Counter

logger = logging.getLogger(__name__)

QUOTAS = {
    "健康核保": 4,
    "案例": 3,
    "财务核保": 3,
    "行政规则": 3,
    "投保规则": 3,
    "一般查询": 2,
}
CROSS_PAGE_QUOTA = 2
TOTAL_QUOTA = 20


def _qa_key(r: dict) -> tuple[str, str]:
    return (r.get("chunk_id") or "", r.get("question") or "")


def _is_cross_page(r: dict) -> bool:
    page = r.get("page")
    if page is None:
        ps, pe = r.get("page_start"), r.get("page_end")
        if ps is None or pe is None:
            return False
        return ps != pe
    return isinstance(page, str) and "-" in page


def _category(r: dict) -> str:
    return r.get("category_simp") or r.get("category") or ""


def stratified_sample(rows: list[dict], random_state: int = 42) -> list[dict]:
    rng = random.Random(random_state)

    picked: dict[tuple[str, str], dict] = {}

    # 1. 跨页 quota 优先（独立于分类配额）
    cross_pool = [r for r in rows if _is_cross_page(r)]
    rng.shuffle(cross_pool)
    cross_picks = cross_pool[:CROSS_PAGE_QUOTA]
    for r in cross_picks:
        picked[_qa_key(r)] = r
    cross_keys = {_qa_key(r) for r in cross_picks}
    logger.info("cross-page picks: %d (pool size %d)", len(cross_picks), len(cross_pool))

    # 2. 各分类按 quota 抽，跨页选中不算分类配额（独立计数）
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(_category(r), []).append(r)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    for cat, want in QUOTAS.items():
        # 不计入跨页选中的；从该分类剩余里抽 want 条
        added = 0
        for r in by_cat.get(cat, []):
            if _qa_key(r) in picked:
                continue
            picked[_qa_key(r)] = r
            added += 1
            if added == want:
                break
        if added < want:
            logger.warning("category %s short: want %d got %d", cat, want, added)

    # 3. 总数补齐到 20（仅当某分类 QA 不足造成总数 < 20 时触发）
    if len(picked) < TOTAL_QUOTA:
        remaining = [r for r in rows if _qa_key(r) not in picked]
        rng.shuffle(remaining)
        for r in remaining:
            if len(picked) >= TOTAL_QUOTA:
                break
            picked[_qa_key(r)] = r

    sampled = list(picked.values())
    if len(sampled) > TOTAL_QUOTA:
        rng.shuffle(sampled)
        sampled = sampled[:TOTAL_QUOTA]

    _warn_category_deviation(sampled, cross_keys)
    _warn_chunk_diversity(sampled, rows)
    return sampled


def _warn_category_deviation(sampled: list[dict], cross_keys: set[tuple[str, str]]) -> None:
    """实际分类数与预期配额不符时打印 warning，说明跨页 QA 落入此分类的数量。"""
    actual = Counter(_category(r) for r in sampled)
    cross_per_cat = Counter(_category(r) for r in sampled if _qa_key(r) in cross_keys)
    deviations = []
    for cat, want in QUOTAS.items():
        got = actual.get(cat, 0)
        if got != want:
            cross_n = cross_per_cat.get(cat, 0)
            deviations.append((cat, got, want, cross_n))
    if not deviations:
        return
    print()
    for cat, got, want, cross_n in deviations:
        if cross_n:
            reason = f"跨页 QA 落入此分类 {cross_n} 条"
        elif got < want:
            reason = "该分类 QA 总数不足"
        else:
            reason = "其他原因（补齐）"
        print(f"⚠ 抽样偏差：{cat} 抽到 {got} 条（预期 {want} 条），原因：{reason}")


def _warn_chunk_diversity(sampled: list[dict], all_rows: list[dict]) -> None:
    """某 chunk_id 在样本中出现 ≥3 次时打印 warning，标出该 chunk 所属分类的全集稀疏度。"""
    cid_cnt = Counter(r.get("chunk_id") for r in sampled)
    over = [(cid, n) for cid, n in cid_cnt.items() if n >= 3]
    if not over:
        return
    # 按 chunk_id 计算所属分类全集的 chunk 多样性
    cat_to_chunks: dict[str, set[str]] = {}
    for r in all_rows:
        cat_to_chunks.setdefault(_category(r), set()).add(r.get("chunk_id") or "")
    print()
    for cid, n in over:
        # 找 sample 里此 chunk 涉及的分类
        cats_in_sample = sorted(set(_category(r) for r in sampled if r.get("chunk_id") == cid))
        diag_parts = []
        for cat in cats_in_sample:
            chunks_in_cat = len(cat_to_chunks.get(cat, set()))
            qa_in_cat = sum(1 for r in all_rows if _category(r) == cat)
            diag_parts.append(f"{cat} 全集 {qa_in_cat} 条 QA / 分布在 {chunks_in_cat} 个 chunk")
        diag = "; ".join(diag_parts)
        short = (cid or "").split("_")[-1]
        print(f"⚠ chunk 多样性偏差：...{short} 在 sample 中出现 {n} 次")
        print(f"  原因：{diag}，约束不到 chunk 多样性")
