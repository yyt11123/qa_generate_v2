"""断点续传 + 失败记录。按输入文件名隔离。"""
import json
import logging
import os
import tempfile
import traceback
from pathlib import Path

from config import FAILED_DIR, OUTPUT_DIR, PROGRESS_DIR

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, obj: dict) -> None:
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


class ProgressTracker:
    """每个 chunk 处理完立即 flush。中间产物 jsonl 也由本类管理（追加写）。

    qa_jsonl_path 优先用调用方传入的显式路径（一般来自 config.INTERMEDIATE_QA_JSONL）；
    若未传入，则按 input_basename 派生（dry-run / 多文件批处理时常用此分支）。
    """

    def __init__(self, input_basename: str, qa_jsonl_path: str | Path | None = None):
        self.basename = input_basename
        self.progress_path = PROGRESS_DIR / f"{input_basename}.progress.json"
        self.failed_path = FAILED_DIR / f"{input_basename}.failed.json"
        if qa_jsonl_path is not None:
            self.qa_jsonl_path = Path(qa_jsonl_path)
        else:
            self.qa_jsonl_path = OUTPUT_DIR / f"{input_basename}.qa.jsonl"

        self.done: set[str] = set()
        self.skipped: set[str] = set()
        self.failed: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.progress_path.exists():
            with self.progress_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            self.done = set(obj.get("done", []))
            self.skipped = set(obj.get("skipped", []))
            logger.info(
                "loaded progress: done=%d skipped=%d (file=%s)",
                len(self.done), len(self.skipped), self.progress_path,
            )
        if self.failed_path.exists():
            with self.failed_path.open("r", encoding="utf-8") as f:
                self.failed = json.load(f)
            logger.info("loaded failed: %d entries", len(self.failed))

    def is_processed(self, chunk_id: str) -> bool:
        return chunk_id in self.done or chunk_id in self.skipped

    def mark_done(self, chunk_id: str, qa_pairs: list[dict], chunk: dict) -> None:
        self.done.add(chunk_id)
        self.failed.pop(chunk_id, None)
        self._append_qa(chunk_id, qa_pairs, chunk)
        self._flush()

    def mark_skipped(self, chunk_id: str) -> None:
        self.skipped.add(chunk_id)
        self.failed.pop(chunk_id, None)
        self._flush()

    def mark_failed(self, chunk_id: str, exc: BaseException) -> None:
        self.failed[chunk_id] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        self._flush()

    def _append_qa(self, chunk_id: str, qa_pairs: list[dict], chunk: dict) -> None:
        self.qa_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.qa_jsonl_path.open("a", encoding="utf-8") as f:
            for qa in qa_pairs:
                row = {
                    "chunk_id": chunk_id,
                    "source": chunk.get("source", ""),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "category": qa.get("category"),
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "supporting_facts": qa.get("supporting_facts"),
                    "type": qa.get("type"),
                    "_overlap": qa.get("_overlap"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _flush(self) -> None:
        _atomic_write_json(
            self.progress_path,
            {"done": sorted(self.done), "skipped": sorted(self.skipped)},
        )
        _atomic_write_json(self.failed_path, self.failed)

    def reset_intermediate(self) -> None:
        """dry-run / 重跑用：清掉中间 jsonl，避免重复 QA 行。"""
        if self.qa_jsonl_path.exists():
            self.qa_jsonl_path.unlink()

    def summary(self) -> dict:
        return {
            "done": len(self.done),
            "skipped_empty": len(self.skipped),
            "failed": len(self.failed),
        }
