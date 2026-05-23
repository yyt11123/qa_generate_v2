"""文件级 manifest：跟踪 10 个输入文件的处理状态。

【数据格式】
{
  "files": {
    "<basename>": {
      "status": "done" | "failed" | "pending",
      "started_at": "...",
      "finished_at": "...",
      "stats": {
        "n_chunks": int,           # 输入 chunks 数
        "n_qa": int,               # 实际生成的 QA 行数
        "elapsed_sec": float,      # 单文件耗时
        "prompt_tokens": int,      # qwen-plus chat_json 累计 prompt tokens
        "completion_tokens": int,  # qwen-plus chat_json 累计 completion tokens
        "n_llm_calls": int         # chat_json 成功调用次数
      },
      "error": "..."            # 仅 failed 状态有
    },
    ...
  },
  "updated_at": "..."
}

【幂等性】
- 已 done 的文件 is_done() 返回 True，batch_runner 直接跳过
- failed 文件可重跑（mark_failed 不阻止下次 mark_done 覆盖）
- 写入用 atomic replace（同 progress.py 思路），避免半写文件破坏 manifest

【关于 token 字段】
只追踪 qwen-plus 的 chat_json（QA 生成）token 用量；embedding API 的 usage 字段
格式不一，且成本占比小，本实现不追踪 embedding tokens。
token 累加器在 qa_generator/llm_client.py 模块级，batch_runner 每个文件 reset 一次。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

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


class BatchManifest:
    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path)
        self.data: dict = {"files": {}, "updated_at": ""}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.data = json.load(f)
            if "files" not in self.data:
                self.data["files"] = {}
            logger.info(
                "loaded batch manifest: %d entries (%s)",
                len(self.data["files"]), self.path,
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("failed to load manifest %s, starting fresh: %s", self.path, e)
            self.data = {"files": {}, "updated_at": ""}

    def is_done(self, basename: str) -> bool:
        return self.data["files"].get(basename, {}).get("status") == "done"

    def get_status(self, basename: str) -> str:
        return self.data["files"].get(basename, {}).get("status", "pending")

    def mark_started(self, basename: str) -> None:
        entry = self.data["files"].setdefault(basename, {})
        entry["status"] = "running"
        entry["started_at"] = datetime.now().isoformat(timespec="seconds")
        entry.pop("error", None)
        self._flush()

    def mark_done(self, basename: str, stats: dict) -> None:
        entry = self.data["files"].setdefault(basename, {})
        entry["status"] = "done"
        entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        entry["stats"] = {
            "n_chunks": int(stats.get("n_chunks", 0)),
            "n_qa": int(stats.get("n_qa", 0)),
            "elapsed_sec": float(stats.get("elapsed_sec", 0.0)),
            "prompt_tokens": int(stats.get("prompt_tokens", 0)),
            "completion_tokens": int(stats.get("completion_tokens", 0)),
            "n_llm_calls": int(stats.get("n_llm_calls", 0)),
        }
        entry.pop("error", None)
        self._flush()

    def mark_failed(self, basename: str, error: str) -> None:
        entry = self.data["files"].setdefault(basename, {})
        entry["status"] = "failed"
        entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        entry["error"] = error
        self._flush()

    def get_summary(self) -> dict:
        files = self.data["files"]
        done = [b for b, e in files.items() if e.get("status") == "done"]
        failed = [b for b, e in files.items() if e.get("status") == "failed"]
        running = [b for b, e in files.items() if e.get("status") == "running"]
        total_qa = sum(
            (files[b].get("stats") or {}).get("n_qa", 0) for b in done
        )
        total_chunks = sum(
            (files[b].get("stats") or {}).get("n_chunks", 0) for b in done
        )
        total_elapsed = sum(
            (files[b].get("stats") or {}).get("elapsed_sec", 0.0) for b in done
        )
        total_prompt_tokens = sum(
            (files[b].get("stats") or {}).get("prompt_tokens", 0) for b in done
        )
        total_completion_tokens = sum(
            (files[b].get("stats") or {}).get("completion_tokens", 0) for b in done
        )
        total_llm_calls = sum(
            (files[b].get("stats") or {}).get("n_llm_calls", 0) for b in done
        )
        return {
            "done": done,
            "failed": failed,
            "running": running,
            "n_done": len(done),
            "n_failed": len(failed),
            "total_qa": total_qa,
            "total_chunks": total_chunks,
            "total_elapsed_sec": total_elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_llm_calls": total_llm_calls,
        }

    def _flush(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write_json(self.path, self.data)
