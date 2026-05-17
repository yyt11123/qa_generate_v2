"""单文件 QA 生成入口。

Usage:
    python main.py                                       # 默认读 config.INPUT_FILE → config.OUTPUT_XLSX
    python main.py --limit 5 --dry-run                   # 前 5 个 chunks dry-run
    python main.py --input inputs/foo.jsonl --output output/foo.xlsx   # 显式覆盖
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    FAIL_THRESHOLD,
    INPUT_FILE,
    INTERMEDIATE_QA_JSONL,
    LOG_DIR,
    OUTPUT_XLSX,
    TOP_K,
)
from qa_generator.chunk_loader import load_chunks
from qa_generator.embedder import index_chunks
from qa_generator.exporter import export_to_xlsx
from qa_generator.progress import ProgressTracker
from qa_generator.qa_extractor import generate_qa_for_chunk
from qa_generator.retriever import query_neighbors


def setup_logging(input_basename: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{input_basename}_{ts}.log"

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG QA test set generator (single-file mode).")
    p.add_argument("--input", default=INPUT_FILE, help=f"path to chunks jsonl (default {INPUT_FILE})")
    p.add_argument("--output", default=OUTPUT_XLSX, help=f"path to output xlsx (default {OUTPUT_XLSX})")
    p.add_argument("--limit", type=int, default=0, help="only process first N chunks (0 = all)")
    p.add_argument("--top-k", type=int, default=TOP_K, help=f"neighbor count (default {TOP_K})")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="dry-run: ignore prior progress, reset intermediate jsonl, write to a separate xlsx",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> int:
    load_dotenv()

    input_path = Path(args.input)
    input_basename = input_path.stem
    if args.dry_run:
        input_basename += ".dryrun"

    log_path = setup_logging(input_basename)
    logger = logging.getLogger("main")
    logger.info("=== run start | input=%s output=%s top_k=%d limit=%d dry_run=%s ===",
                args.input, args.output, args.top_k, args.limit, args.dry_run)
    logger.info("log file: %s", log_path)

    chunks = load_chunks(input_path)
    if args.limit > 0:
        chunks = chunks[: args.limit]
        logger.info("limited to first %d chunks", len(chunks))

    stats = index_chunks(chunks)
    logger.info("indexing done: %s", stats)

    tracker = ProgressTracker(
        input_basename,
        qa_jsonl_path=None if args.dry_run else INTERMEDIATE_QA_JSONL,
    )
    if args.dry_run:
        tracker.reset_intermediate()
        tracker.done.clear()
        tracker.skipped.clear()
        tracker.failed.clear()
        tracker._flush()
        logger.info("dry-run: cleared progress + intermediate qa.jsonl")

    pending = [c for c in chunks if not tracker.is_processed(c["chunk_id"])]
    logger.info("to process: %d / %d", len(pending), len(chunks))

    fail_count = 0
    qa_total = 0

    for chunk in tqdm(pending, desc="generating QA", unit="chunk"):
        cid = chunk["chunk_id"]
        try:
            t0 = time.time()
            neighbors = query_neighbors(chunk, k=args.top_k)
            neighbor_ids = [n["chunk_id"] for n in neighbors]
            logger.info("[%s] neighbors=%s", cid, neighbor_ids)

            qa_pairs = generate_qa_for_chunk(chunk, neighbors)
            elapsed = time.time() - t0

            if not qa_pairs:
                tracker.mark_skipped(cid)
                logger.info("[%s] skipped (empty qa) in %.2fs", cid, elapsed)
            else:
                tracker.mark_done(cid, qa_pairs, chunk)
                qa_total += len(qa_pairs)
                logger.info("[%s] generated %d qa in %.2fs", cid, len(qa_pairs), elapsed)
        except Exception as e:
            fail_count += 1
            tracker.mark_failed(cid, e)
            logger.exception("[%s] FAILED (%d): %s", cid, fail_count, e)
            if fail_count >= FAIL_THRESHOLD:
                logger.error(
                    "failure count reached threshold %d, stopping. failed file: %s",
                    FAIL_THRESHOLD, tracker.failed_path,
                )
                break

    summary = tracker.summary()
    logger.info("=== run summary: %s qa_pairs_total=%d ===", summary, qa_total)

    n_rows = export_to_xlsx(tracker.qa_jsonl_path, args.output)
    logger.info("xlsx exported: %s (%d rows)", args.output, n_rows)

    if fail_count >= FAIL_THRESHOLD:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
