import os, math
from typing import List, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from sentence_transformers import SentenceTransformer

QDRANT_URL        = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_rh_chunks")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

_qc = None
_model = None

def get_qdrant() -> QdrantClient:
    global _qc
    if _qc is None:
        _qc = QdrantClient(url=QDRANT_URL)
    return _qc

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        _model.encode(["warmup"], normalize_embeddings=True)
    return _model

def embed(text: str) -> List[float]:
    v = get_model().encode([text], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32).tolist()

def _build_filter(source_filter: Optional[str]):
    if not source_filter:
        return None
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])

def search(
    query: str,
    top_k: int = 8,
    source_filter: Optional[str] = None,
    use_rerank: bool = False,
    candidate_k: Optional[int] = None,
) -> List[dict]:
    """
    Search for relevant documents using vector similarity and optional reranking.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        source_filter: Filter by document source (e.g., 'travail-emploi')
        use_rerank: Whether to apply cross-encoder reranking
        candidate_k: Number of candidates for initial retrieval (auto-calculated if None)
        
    Returns:
        List of search results with scores and payloads
    """
    qc = get_qdrant()
    qv = embed(query)

    # pull more candidates if reranking
    k = candidate_k or max(32, top_k)
    if use_rerank:
        k = max(k, top_k + 8)

    hits = qc.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qv,
        limit=k,
        query_filter=_build_filter(source_filter),
        search_params=SearchParams(hnsw_ef=128),
        with_payload=True,
        with_vectors=False,
    )

    # Optional rerank
    if use_rerank:
        try:
            from rag_app.reranker import rerank as _rerank
            hits = _rerank(
                query,
                hits,
                top_k=top_k,
                text_field="text",
                device=os.getenv("RERANK_DEVICE", "cpu"),
                batch_size=int(os.getenv("RERANK_BATCH", "16")),
            )
        except Exception as e:
            # fall back gracefully
            print(f"[rerank] disabled (error: {e})")
            hits = hits[:top_k]
    else:
        hits = hits[:top_k]

    # Normalize into dicts the UI can consume
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "id": str(getattr(h, "id", "")),
            "score": float(getattr(h, "score", math.nan)) if hasattr(h, "score") else None,
            "payload": {
                **p,
                # keep rerank_score if the reranker wrote it
                "rerank_score": p.get("rerank_score"),
            }
        })
    return out