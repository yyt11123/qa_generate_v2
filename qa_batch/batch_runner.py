"""批量主循环：扫描 inputs/batch/*.jsonl → 串行处理 → 汇总输出。

【流程】
  1. 扫描 BATCH_INPUT_DIR 下的 *.jsonl，按文件名排序
  2. 用 BatchManifest 过滤已 done 的文件
  3. 串行：对每个文件
     - load_chunks
     - index_chunks(collection_name=CHROMA_COLLECTION_BATCH, scope_by_source=True)
     - 复用单文件流水线: query_neighbors(source_filter=<source>) + generate_qa_for_chunk
     - chunk 失败：跳过该 chunk；累计 ≥ FAIL_THRESHOLD 该文件失败
     - 写 output/per_file/<basename>.xlsx
  4. ≥ BATCH_FAIL_FILE_THRESHOLD 个文件失败：暂停（不再处理后续文件）
  5. 全跑完调用 aggregator 合并 → output/qa_test_all.xlsx

【日志/输出】
  - tqdm: [N/total] basename | chunks=X | QA=Y | elapsed=Zs
  - 单文件 xlsx 用 export_to_xlsx 写到 BATCH_OUTPUT_DIR
  - 单文件 jsonl 中间产物落到 OUTPUT_DIR/<basename>.qa.jsonl（与 v3 共存，前缀不同自然隔离）

【与 v3 隔离】
  - 单独 collection: chunks_batch
  - 单独 manifest:   progress/_batch.manifest.json
  - 单独失败汇总:    failed/_batch.failed.json
  - v3 vector_db / output / progress 文件不会被本流程动到
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from openai import PermissionDeniedError
from tqdm import tqdm

from config import (
    BATCH_AGGREGATED_XLSX,
    BATCH_FAIL_FILE_THRESHOLD,
    BATCH_FAILED_LOG,
    BATCH_INPUT_DIR,
    BATCH_MANIFEST,
    BATCH_OUTPUT_DIR,
    CHROMA_COLLECTION_BATCH,
    FAIL_THRESHOLD,
    LOG_DIR,
    OUTPUT_DIR,
    TOP_K,
)
from qa_batch.manifest import BatchManifest
from qa_generator.chunk_loader import load_chunks
from qa_generator.embedder import index_chunks
from qa_generator.exporter import export_to_xlsx
from qa_generator.llm_client import get_token_stats, reset_token_stats
from qa_generator.progress import ProgressTracker
from qa_generator.qa_extractor import generate_qa_for_chunk
from qa_generator.retriever import query_neighbors


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


# qwen-plus 计费（CNY/百万 token）
_PRICE_PROMPT_PER_M = 0.8
_PRICE_COMPLETION_PER_M = 2.0

# 单次批跑预算告警阈值（CNY）。超过 → 暂停整批，等用户决定。
COST_ABORT_THRESHOLD_CNY = 15.0


def _cost_cny(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * _PRICE_PROMPT_PER_M + completion_tokens * _PRICE_COMPLETION_PER_M) / 1_000_000


def setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"batch_run_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    return log_path


def discover_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(input_dir.glob("*.jsonl"))


def filter_by_specs(all_inputs: list[Path], specs: list[str]) -> list[Path]:
    """按 specs 顺序保留匹配文件。每个 spec 用「子串包含」匹配 path.name。

    - 0 个匹配 → ValueError
    - 多个匹配 → ValueError（spec 太宽，让用户给更具体的串）
    - 重复 spec 命中同一文件也允许，但只保留第一次（避免同一文件跑两遍）
    返回顺序按 specs 顺序，不是字母顺序——重要：用户传顺序通常意味着想按这个顺序跑。
    """
    chosen: list[Path] = []
    seen: set[str] = set()
    for spec in specs:
        hits = [p for p in all_inputs if spec in p.name]
        if not hits:
            raise ValueError(f"--files spec {spec!r} matched no file")
        if len(hits) > 1:
            names = [p.name for p in hits]
            raise ValueError(f"--files spec {spec!r} matched multiple files: {names}")
        p = hits[0]
        if p.name not in seen:
            chosen.append(p)
            seen.add(p.name)
    return chosen


def _resolve_source(chunks: list[dict], fallback: str) -> str:
    """优先从 chunk metadata 取 source；缺失则用文件名兜底。
    源数据每个 chunk 都带 source（chunk_loader 会校验前几个必填字段，
    但 source 是独立校验项），这里只是双保险。"""
    for c in chunks:
        s = c.get("source")
        if s:
            return s
    return fallback


def process_one_file(
    input_path: Path,
    output_dir: Path,
    top_k: int,
) -> dict:
    """处理单个文件。返回 stats dict（成功/失败都返回，区别在 raised 异常）。

    chunk 失败计数 ≥ FAIL_THRESHOLD 时抛 RuntimeError，由上层标 failed。
    """
    basename = input_path.stem
    logger = logging.getLogger("batch")
    t0 = time.time()
    reset_token_stats()

    chunks = load_chunks(input_path)
    n_chunks = len(chunks)
    if n_chunks == 0:
        ts = get_token_stats()
        return {
            "n_chunks": 0,
            "n_qa": 0,
            "elapsed_sec": time.time() - t0,
            "prompt_tokens": ts["prompt_tokens"],
            "completion_tokens": ts["completion_tokens"],
            "n_llm_calls": ts["n_calls"],
        }

    source_str = _resolve_source(chunks, basename)

    idx_stats = index_chunks(
        chunks,
        collection_name=CHROMA_COLLECTION_BATCH,
        scope_by_source=True,
    )
    logger.info("[%s] index: %s", basename, idx_stats)

    qa_jsonl_path = OUTPUT_DIR / f"{basename}.qa.jsonl"
    tracker = ProgressTracker(basename, qa_jsonl_path=qa_jsonl_path)
    pending = [c for c in chunks if not tracker.is_processed(c["chunk_id"])]
    logger.info("[%s] to process: %d / %d", basename, len(pending), n_chunks)

    fail_count = 0
    qa_total = 0

    for chunk in tqdm(pending, desc=basename[:30], unit="chunk", leave=False):
        cid = chunk["chunk_id"]
        try:
            ts = time.time()
            neighbors = query_neighbors(
                chunk,
                k=top_k,
                collection_name=CHROMA_COLLECTION_BATCH,
                source_filter=source_str,
            )
            qa_pairs = generate_qa_for_chunk(chunk, neighbors)
            elapsed = time.time() - ts

            if not qa_pairs:
                tracker.mark_skipped(cid)
                logger.info("[%s/%s] skipped (empty qa) in %.2fs", basename, cid, elapsed)
            else:
                tracker.mark_done(cid, qa_pairs, chunk)
                qa_total += len(qa_pairs)
                logger.info("[%s/%s] %d qa in %.2fs", basename, cid, len(qa_pairs), elapsed)
        except PermissionDeniedError:
            # 403：配额/权限问题，立即抛给外层中止整批，不计入 chunk fail_count
            logger.error("[%s/%s] 403 from API, aborting batch", basename, cid)
            raise
        except Exception as e:
            fail_count += 1
            tracker.mark_failed(cid, e)
            logger.exception("[%s/%s] FAILED (%d): %s", basename, cid, fail_count, e)
            if fail_count >= FAIL_THRESHOLD:
                raise RuntimeError(
                    f"chunk failures reached {FAIL_THRESHOLD} for {basename}"
                ) from e

    output_xlsx = output_dir / f"{basename}.xlsx"
    n_rows = export_to_xlsx(tracker.qa_jsonl_path, output_xlsx)
    elapsed_total = time.time() - t0
    token_stats = get_token_stats()
    logger.info(
        "[%s] DONE: chunks=%d qa=%d xlsx_rows=%d elapsed=%.1fs llm_calls=%d "
        "prompt_tokens=%d completion_tokens=%d",
        basename, n_chunks, qa_total, n_rows, elapsed_total,
        token_stats["n_calls"], token_stats["prompt_tokens"], token_stats["completion_tokens"],
    )

    return {
        "n_chunks": n_chunks,
        "n_qa": qa_total,
        "elapsed_sec": elapsed_total,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "n_llm_calls": token_stats["n_calls"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch QA generator (10-file mode).")
    p.add_argument("--input-dir", default=BATCH_INPUT_DIR)
    p.add_argument("--output-dir", default=BATCH_OUTPUT_DIR)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument(
        "--limit-files", type=int, default=0,
        help="only process first N files (0 = all). useful for the 3-file pilot.",
    )
    p.add_argument(
        "--files", default="",
        help="comma-separated substrings; only run files whose name contains one. "
             "order matters — files run in spec order, not alphabetical. "
             "mutually exclusive with --limit-files.",
    )
    p.add_argument(
        "--no-aggregate", action="store_true",
        help="skip the final per_file → qa_test_all.xlsx merge step.",
    )
    p.add_argument(
        "--reset-manifest", action="store_true",
        help="ignore prior manifest done state (still keeps per-chunk progress).",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    log_path = setup_logging()
    logger = logging.getLogger("batch")
    logger.info("=== batch run start | log=%s ===", log_path)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(input_dir)
    if not inputs:
        logger.error("no jsonl files under %s", input_dir)
        return 2

    if args.files and args.limit_files:
        logger.error("--files and --limit-files are mutually exclusive")
        return 2
    if args.files:
        specs = [s.strip() for s in args.files.split(",") if s.strip()]
        inputs = filter_by_specs(inputs, specs)
        logger.info("filtered to %d files via --files: %s",
                    len(inputs), [p.name for p in inputs])
    elif args.limit_files > 0:
        inputs = inputs[: args.limit_files]
        logger.info("limited to first %d files", len(inputs))

    manifest = BatchManifest(BATCH_MANIFEST)
    if args.reset_manifest:
        manifest.data = {"files": {}, "updated_at": ""}
        manifest._flush()
        logger.info("manifest reset")

    failed_files: list[dict] = []
    total = len(inputs)
    paused = False
    aborted_403 = False
    aborted_cost = False
    cumulative_prompt = 0
    cumulative_completion = 0

    # 把已 done 文件的累计 token 算进预算（重启续跑时也对得上）
    pre_summary = manifest.get_summary()
    cumulative_prompt += pre_summary["total_prompt_tokens"]
    cumulative_completion += pre_summary["total_completion_tokens"]
    logger.info(
        "starting cumulative tokens (from manifest done files): prompt=%d completion=%d cost=¥%.4f",
        cumulative_prompt, cumulative_completion,
        _cost_cny(cumulative_prompt, cumulative_completion),
    )

    for idx, ip in enumerate(inputs, start=1):
        basename = ip.stem
        if manifest.is_done(basename):
            logger.info("[%d/%d] %s | already done, skipping", idx, total, basename)
            continue

        logger.info("[%d/%d] %s | starting", idx, total, basename)
        manifest.mark_started(basename)
        try:
            stats = process_one_file(ip, output_dir, top_k=args.top_k)
            manifest.mark_done(basename, stats)
            cumulative_prompt += stats["prompt_tokens"]
            cumulative_completion += stats["completion_tokens"]
            running_cost = _cost_cny(cumulative_prompt, cumulative_completion)
            logger.info(
                "[%d/%d] %s | chunks=%d QA=%d elapsed=%.1fs running_cost=¥%.4f",
                idx, total, basename, stats["n_chunks"], stats["n_qa"],
                stats["elapsed_sec"], running_cost,
            )
            if running_cost >= COST_ABORT_THRESHOLD_CNY:
                logger.error(
                    "cumulative cost ¥%.4f reached threshold ¥%.2f, pausing batch run.",
                    running_cost, COST_ABORT_THRESHOLD_CNY,
                )
                aborted_cost = True
                paused = True
                break
        except PermissionDeniedError as e:
            err = f"PermissionDeniedError: {e}"
            manifest.mark_failed(basename, err)
            failed_files.append({"file": basename, "error": err})
            _atomic_write_json(Path(BATCH_FAILED_LOG), failed_files)
            logger.error("[%d/%d] %s | 403 from API, aborting whole batch", idx, total, basename)
            aborted_403 = True
            paused = True
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            manifest.mark_failed(basename, err)
            failed_files.append({"file": basename, "error": err})
            _atomic_write_json(Path(BATCH_FAILED_LOG), failed_files)
            logger.exception("[%d/%d] %s | FAILED: %s", idx, total, basename, err)

            if len(failed_files) >= BATCH_FAIL_FILE_THRESHOLD:
                logger.error(
                    "failed files reached threshold %d, pausing batch run.",
                    BATCH_FAIL_FILE_THRESHOLD,
                )
                paused = True
                break

    summary = manifest.get_summary()
    logger.info("=== batch summary: %s ===", summary)

    if not paused and not args.no_aggregate:
        from qa_batch.aggregator import aggregate
        agg_path = Path(BATCH_AGGREGATED_XLSX)
        n_rows = aggregate(output_dir, agg_path)
        logger.info("aggregated → %s (%d rows)", agg_path, n_rows)

    print()
    print("===== BATCH SUMMARY =====")
    print(f"total files       : {total}")
    print(f"done              : {summary['n_done']}")
    print(f"failed            : {summary['n_failed']}  {summary['failed']}")
    print(f"total chunks      : {summary['total_chunks']}")
    print(f"total QA pairs    : {summary['total_qa']}")
    print(f"total elapsed     : {summary['total_elapsed_sec']:.1f}s")
    print(f"total LLM calls   : {summary['total_llm_calls']}")
    print(
        f"total tokens (prompt/completion): "
        f"{summary['total_prompt_tokens']} / {summary['total_completion_tokens']}  "
        f"(qwen-plus only; embedding tokens not tracked)"
    )
    print(
        f"cost (qwen-plus)  : ¥{_cost_cny(summary['total_prompt_tokens'], summary['total_completion_tokens']):.4f}"
    )
    if aborted_403:
        print("PAUSED: 403 PermissionDenied from API, aborted whole batch")
        return 3
    if aborted_cost:
        print(f"PAUSED: cumulative cost reached ¥{COST_ABORT_THRESHOLD_CNY:.2f}")
        return 4
    if paused:
        print(f"PAUSED: failed >= {BATCH_FAIL_FILE_THRESHOLD}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
