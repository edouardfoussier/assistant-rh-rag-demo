"""
RAG-RH Demo - Streamlit Application

A Retrieval-Augmented Generation application for HR assistance using:
- Qdrant vector database for document retrieval
- Sentence transformers for embeddings
- Cross-encoder reranking (optional)
- LLM-based answer synthesis

Features:
- Multi-source document search (travail-emploi, service-public)
- Configurable retrieval parameters
- Interactive question suggestions
- Citation-linked answers

"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))       # /app/rag_app
_repo_root = os.path.dirname(_here)                      # /app
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# -----------------------------------------------
import time
from collections import Counter
from typing import Optional, List
import random

import streamlit as st
from dotenv import load_dotenv
import numpy as np
from qdrant_client import QdrantClient

from rag_app.retrieval import search
from rag_app.synth import answer as synth_answer 
from rag_app.synth import linkify_citations
from rag_app.synth import ping_models, LLM_BASE_URL, LLM_MODEL
from rag_app.utils_logging import log_search

import warnings
warnings.filterwarnings(
    "ignore",
    message="`search` method is deprecated",
    category=DeprecationWarning,
)

load_dotenv(override=True)

# --- One-time warmup per Streamlit session ---
if "warmed_up" not in st.session_state:
    st.session_state["warmed_up"] = False

def _do_warmup():
    """Prime embedder, reranker, and Qdrant connection."""
    try:
        # 1) embedder: one tiny encode to compile kernels / load weights
        from rag_app.retrieval import get_model, get_qdrant, QDRANT_COLLECTION, embed
        _ = get_model().encode(["warmup"], normalize_embeddings=True)

        # 2) Qdrant: quick no-op-ish call to open HTTP/GRPC, warm HNSW
        qc: QdrantClient = get_qdrant()
        # cheap vector (zeros ok since we just want to prime connection)
        dummy = np.random.randn(VECTOR_DIMENSION).astype(np.float32)
        dummy /= np.linalg.norm(dummy) + 1e-12
        dummy = dummy.tolist()
        try:
            qc.search(collection_name=QDRANT_COLLECTION, query_vector=dummy, limit=1)
        except Exception:
            # If your collection enforces non-zero vector norms, ignore errors
            pass

    finally:
        st.session_state["warmed_up"] = True
        
# ------------------------------------------------------------
# Demo questions by source
# ------------------------------------------------------------
DEMO_QUESTIONS = {
    "travail-emploi": [
        "Quelles sont les conditions pour bénéficier du contrat d’apprentissage ?",
        "Quelles aides existent pour l’embauche d’un demandeur d’emploi en contrat de professionnalisation ?",
        "Comment fonctionne l’aide exceptionnelle pour l’embauche d’un alternant ?",
        "Quelles sont les obligations de l’employeur concernant la déclaration préalable à l’embauche (DPAE) ?",
        "Quelles sont les obligations de l’employeur pour le télétravail ?",
	    "Comment fonctionne la période d’essai dans un contrat de professionnalisation ?"
    ],
    "service-public": [
        "Quels sont les droits à congés pour un agent contractuel de la fonction publique ?",
        "Comment déclarer l’embauche d’un salarié auprès de l’URSSAF (DPAE) ?",
        "Quelles démarches effectuer pour obtenir une attestation d’employeur destinée à Pôle emploi ?",
        "Quels sont les droits des agents publics en matière de congé parental ?",
        "Comment déclarer un arrêt maladie dans la fonction publique ?",
        "Quelles démarches un agent doit-il effectuer pour demander sa retraite ?"
    ],
}

# ------------------------------------------------------------
# Page & styles
# ------------------------------------------------------------
st.set_page_config(page_title="RAG-RH Demo", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.badge {
  display:inline-block; padding:2px 8px; border-radius:999px;
  font-size:12px; font-weight:600; line-height:1; color:white;
  vertical-align:middle; margin-right:8px;}
.badge-te   { background:#2563eb; }  /* bleu  (travail-emploi) */
.badge-sp   { background:#059669; }  /* vert  (service-public) */
.badge-unk  { background:#6b7280; }  /* gris  (unknown/autre) */
.title-line { display:flex; align-items:center; gap:.5rem; }
.title-line a { text-decoration:none; }
button[kind="secondary"] { white-space: normal; }
a.chunk-link {
    color: inherit !important;       
    text-decoration: none !important; }
a.chunk-link:hover {
    text-decoration: underline; }
.stToast {
    top: auto !important;
    bottom: 20px !important;
    right: 20px !important; }
[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    height: 100%; }
.sidebar-content {
    flex: 1 1 auto;
    overflow-y: auto; }
.sidebar-footer {
    flex-shrink: 0;
    margin-top: auto; 
    font-size: 0.8em;
    padding: 8px 6px;
    text-align: center; }
.sidebar-footer a { text-decoration: none; color: inherit; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------
for key, default in {
    "query": "",
    "last_query": "",
    "last_hits": [],
    "last_answer": "",
    "last_latency_ms": 0.0,
    "debug": False,
    "sug_keys": None,
    "sugs": [],
    "trigger_search": False,
    "first_search_done": False,
    "warm_toast_shown": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
        

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _payload_of(hit):
    """Extract payload from search hit, handling both dict and object formats."""
    if isinstance(hit, dict):
        return hit.get("payload", hit)
    return (getattr(hit, "payload", None) or {})

def source_badge_html(src: Optional[str]) -> str:
    s = (src or "").lower()
    if s == "travail-emploi":
        cls, label = "badge-te", "travail-emploi"
    elif s == "service-public":
        cls, label = "badge-sp", "service-public"
    else:
        cls, label = "badge-unk", (s or "unknown")
    return f'<span class="badge {cls}">{label}</span>'

def count_by_source(hits) -> Counter:
    """Count search results by source type."""
    c = Counter()
    for h in hits or []:
        pl = _payload_of(h)          
        src = (pl.get("source") or "unknown").lower()
        c[src] += 1
    return c
    
def generate_suggestions(source_filter: str | None) -> list[str]:
    """Generate random question suggestions based on source filter."""
    te = (DEMO_QUESTIONS.get("travail-emploi") or [])
    sp = (DEMO_QUESTIONS.get("service-public") or [])

    if source_filter == "travail-emploi":
        return random.sample(te, k=min(3, len(te)))
    if source_filter == "service-public":
        return random.sample(sp, k=min(3, len(sp)))

    # (Tous) → sample across both pools and prefix labels
    pool = [f"[travail-emploi] {q}" for q in te] + [f"[service-public] {q}" for q in sp]
    return random.sample(pool, k=min(3, len(pool)))


def _use_example(q: str):
    """Set query and trigger search for example question."""
    st.session_state["query"] = q
    st.session_state["trigger_search"] = True
    
def _trigger_search():
    """Single source of truth for programmatic searches."""
    st.session_state["trigger_search"] = True


# ------------------------------------------------------------
# UI — Header
# ------------------------------------------------------------
st.title("🔎 Assistant RH — RAG Demo")

if not st.session_state["warmed_up"]:
    if not st.session_state["warm_toast_shown"]:
        st.toast(" Initialisation des modèles...", icon="⚡")
        st.session_state["warm_toast_shown"] = True    

# ------------------------------------------------------------
# Sidebar — Controls
# ------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

st.sidebar.header("Recherche")
candidate_k = st.sidebar.slider("Nb candidats (retriever)", 10, 32, 16, 1)
use_rerank = st.sidebar.checkbox("Activer le rerank (cross-encoder)", value=False)
top_k = st.sidebar.slider("Top-K (résultats finaux)", 3, 20, 8, 1)

st.sidebar.header("Filtres")
sources = ["(Tous)", "travail-emploi", "service-public"]
source_choice = st.sidebar.selectbox("Choisissez un corpus", sources, index=0)
source_filter = None if source_choice == "(Tous)" else source_choice

if "suggestions" not in st.session_state:
    # génère en fonction du filtre courant UNIQUEMENT au premier chargement
    st.session_state["suggestions"] = generate_suggestions(source_filter)
    st.session_state["suggestions_filter_at_init"] = source_filter 

sugs = st.session_state["suggestions"]

st.sidebar.markdown("""
<div class="sidebar-footer">
    ℹ️ Public information sourced from
    <a href="https://huggingface.co/datasets/AgentPublic/Mediatech" target="_blank">
        AgentPublic's Mediatech on HF 🤗
    </a><br>
    <span style="font-size:0.75em;">
        RAG project portfolio by <b>Edouard FOUSSIER</b>
    </span>
</div>
</div> <!-- ferme sidebar-content -->
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Main input + suggestions
# ------------------------------------------------------------
left, right = st.columns([3, 1])

with left:
    with st.form(key="search_form", clear_on_submit=False):
        query = st.text_input(
            "Votre question RH",
            key="query",
            placeholder="Posez votre question à notre assistant",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Rechercher", type="primary")

    # One unified trigger for search:
    do_search = submitted or st.session_state.pop("trigger_search", False)


    # ------------------------------------------------------------
    # Search (updates session state)
    # ------------------------------------------------------------
  
    if do_search and query.strip():
        with st.spinner("🔎 Recherche en cours…"):
            t0 = time.time()
            hits_now = search(
                query=query,
                top_k=top_k,
                source_filter=source_filter,
                use_rerank=use_rerank,
                candidate_k=candidate_k,
            )
            dt = time.time() - t0
            if source_filter:
                allowed = {source_filter} if isinstance(source_filter, str) else set(source_filter)
                hits_now = [h for h in hits_now
                    if _payload_of(h).get("source") in allowed]

        st.session_state["last_query"] = query
        st.session_state["last_hits"] = hits_now
        st.session_state["last_answer"] = ""
        st.session_state["last_latency_ms"] = dt * 1000.0
        st.session_state["last_latency_s"] = dt

        log_search(
            query=query,
            hits=hits_now,
            latency_ms=dt * 1000.0,
            use_rerank=use_rerank,
            candidate_k=candidate_k,
            source_filter=source_filter,
            top_k=top_k,
        )

    # ------------------------------------------------------------
    # Results section (always rendered from session)
    # ------------------------------------------------------------
    hits = st.session_state.get("last_hits", [])
    if hits:
        st.write(f"🔎 **Top {len(hits)} résultats** — ⏱ {st.session_state.get('last_latency_s',0):.2f} s")

        counts = count_by_source(hits)
        if counts:
            order = ["travail-emploi", "service-public"] + [k for k in counts.keys() if k not in ("travail-emploi", "service-public")]
            cols = st.columns(len(order))
            for col, src in zip(cols, order):
                val = counts.get(src, 0)
                label = "Travail-emploi" if src == "travail-emploi" else ("Service-public" if src == "service-public" else src.title())
                col.metric(label, val)

        for i, h in enumerate(hits, 1):
            pl = _payload_of(h)
            src  = pl.get("source")
            tit  = (pl.get("title") or "").strip() or "(Sans titre)"
            url  = (pl.get("url") or "").strip()
            text = pl.get("text") or pl.get("chunk_text") or ""

            # score compatible dict / objet
            retr_score = (h.get("score") if isinstance(h, dict) else getattr(h, "score", None))
            rerank_score = pl.get("rerank_score")

            parts = []
            if isinstance(retr_score, (int, float)):
                parts.append(f"retriever={retr_score:.3f}")
            if isinstance(rerank_score, (int, float)):
                parts.append(f"**rerank={rerank_score:.3f}**")
            suffix = f"  •  {'  |  '.join(parts)}" if parts else ""

            badge = source_badge_html(src)
            link_html = f'<a href="{url}" target="_blank">{tit}</a>' if url else tit
            head = f'<div class="title-line">{badge}{link_html}</div>'

            with st.expander(f"#{i}  {tit[:80]}{suffix}"):
                st.markdown(head, unsafe_allow_html=True)
                meta_parts = []
                if pl.get("date"):
                    meta_parts.append(str(pl["date"]))


                st.write(text[:700] + ("…" if len(text) > 700 else ""))


    # ------------------------------------------------------------
    # Standalone Synthèse section
    # ------------------------------------------------------------
    st.subheader("📝 Réponse")
    if not hits:
        st.caption("Faites d’abord une recherche pour activer la synthèse.")
    if hits:
        if st.session_state.get("last_hits"):
            if st.button("Synthétiser une réponse",  type="primary"):
                with st.spinner("Synthèse en cours…"):
                    try:
                        ans = synth_answer(
                            st.session_state["last_query"], 
                            st.session_state["last_hits"][:5]
                        )
                        st.session_state["last_answer"] = ans
                    except Exception as e:
                        st.session_state["last_answer"] = f"❌ Erreur lors de la synthèse : {e}"

    # Show answer if available
    if st.session_state.get("last_answer"):
        passages = (st.session_state.get("last_hits") or [])[:5]
        ans_linked = linkify_citations(st.session_state["last_answer"], passages)
        st.markdown(ans_linked, unsafe_allow_html=False)  # render [n](url) as clickable links 
        
with right:
    st.write("**Suggestions**")

    if sugs:
        # 3 boutons empilés, largeur fixe conteneur
        for i in range(3):
            if i < len(sugs):
                label = sugs[i]
                clean = label.split("] ", 1)[1] if label.startswith("[") else label
                st.button(
                    clean,
                    key=f"sug_{i}",
                    use_container_width=True,
                    on_click=lambda q=clean: st.session_state.update(query=q, trigger_search=True),
                )
    if st.button("↻ Shuffle"):
        base_filter = st.session_state.get("suggestions_filter_at_init")
        st.session_state["suggestions"] = generate_suggestions(base_filter)
        