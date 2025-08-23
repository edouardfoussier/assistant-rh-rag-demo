# rag_app/utils_logging.py
from __future__ import annotations
import csv, os, time, socket
from datetime import datetime, timezone
from typing import Any
import pytz


# Where to write the CSV (override with env var if you like)
LOG_PATH = os.getenv("SEARCH_LOG_PATH", "/app/logs/search_log.csv")

paris = pytz.timezone("Europe/Paris")


def _ensure_header(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "ts_iso", "ts_unix", "host",
                "query", "top_k", "candidate_k", "source_filter",
                "use_rerank", "latency_ms",
                "top1_title", "n_hits",
            ])

def _payload_of(hit: Any) -> dict:
    if isinstance(hit, dict):
        return hit.get("payload", hit) or {}
    return (getattr(hit, "payload", None) or {})

def _top1_title(hits: list[Any] | None) -> str:
    if not hits:
        return ""
    return (_payload_of(hits[0]).get("title") or "").strip()

def log_search(
    query: str,
    hits: list[Any] | None,
    latency_ms: float | int | None,
    use_rerank: bool | None,
    candidate_k: int | None,
    source_filter: str | list[str] | None,
    top_k: int | None = None,
) -> None:
    """Append one search row to CSV. Keep this module dependency-free (no streamlit)."""
    _ensure_header(LOG_PATH)
    row = [
        datetime.now(paris).isoformat(),
        f"{time.time():.3f}",
        socket.gethostname(),
        query,
        (top_k if top_k is not None else ""),
        (candidate_k if candidate_k is not None else ""),
        (",".join(source_filter) if isinstance(source_filter, list) else (source_filter or "")),
        int(bool(use_rerank)) if use_rerank is not None else "",
        f"{float(latency_ms):.1f}" if isinstance(latency_ms, (int, float)) else "",
        _top1_title(hits),
        len(hits or []),
    ]
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

__all__ = ["log_search", "LOG_PATH"]