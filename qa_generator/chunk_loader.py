"""读 jsonl chunks 文件。"""
import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("chunk_id", "source", "content", "content_with_context")


def load_chunks(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"chunks file not found: {path}")
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno} is not valid JSON: {e}") from e
            missing = [k for k in REQUIRED_FIELDS if k not in obj]
            if missing:
                raise ValueError(f"line {lineno} (chunk_id={obj.get('chunk_id')}) missing fields: {missing}")
            chunks.append(obj)
    logger.info("loaded %d chunks from %s", len(chunks), path)
    return chunks


def iter_chunks(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
