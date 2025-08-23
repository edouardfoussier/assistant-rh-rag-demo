from typing import List, Dict, Any
from sentence_transformers.cross_encoder import CrossEncoder

_rerank_model = None
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 8

def _get_model(device: str = "cpu"):
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(RERANK_MODEL, device=device)
    return _rerank_model

def rerank(
    query: str,
    hits: List,
    top_k: int = DEFAULT_TOP_K,
    text_field: str = "text",
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Any]:
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        query: The search query
        hits: List of search results with payload containing text
        top_k: Number of top results to return
        text_field: Field name in payload containing text to rerank
        device: Device to run model on ('cpu', 'cuda', etc.)
        batch_size: Batch size for model prediction
        
    Returns:
        List of reranked hits sorted by relevance score
    """
    if not hits:
        return hits
    try:
        model = _get_model(device)
        pairs = []
        for h in hits:
            p = h.payload or {}
            text = p.get(text_field) or p.get("chunk_text") or ""
            pairs.append((query, text))

        scores = model.predict(pairs, batch_size=batch_size)
        # write score into payload for UI
        for h, s in zip(hits, scores):
            if h.payload is None:
                h.payload = {}
            h.payload["rerank_score"] = float(s)

        # sort by rerank_score desc and return top_k
        hits_sorted = sorted(hits, key=lambda x: x.payload.get("rerank_score", 0.0), reverse=True)
        return hits_sorted[:top_k]
    except Exception as e:
        print(f"Warning: Reranking failed, returning original results: {e}")
        return hits[:top_k]  # Fallback to original order