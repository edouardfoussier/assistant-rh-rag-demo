import os
import re
import json
import textwrap
import requests
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(override=True)

# ---- Config (env) ----
LLM_BASE_URL    = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

_MONTHS_FR = {
    1:"janvier", 2:"février", 3:"mars", 4:"avril", 5:"mai", 6:"juin",
    7:"juillet", 8:"août", 9:"septembre", 10:"octobre", 11:"novembre", 12:"décembre"
}
def today_fr():
    d = date.today()
    return f"{d.day} {_MONTHS_FR[d.month]} {d.year}"

def _get_api_key() -> Optional[str]:
    return os.getenv("LLM_API_KEY")

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    k = _get_api_key()
    if k:
        h["Authorization"] = f"Bearer {k}"
    return h

def _payload(hit: Any) -> Dict[str, Any]:
    """
    Return the payload dict whether hit is a Qdrant 'ScoredPoint' (has .payload)
    or already a dict with a 'payload' key (or flattened fields).
    """
    if isinstance(hit, dict):
        if "payload" in hit and isinstance(hit["payload"], dict):
            return hit["payload"]
        return hit
    return getattr(hit, "payload", {}) or {}

# ---- Utilities ----
def ping_models() -> Tuple[bool, str]:
    url = f"{LLM_BASE_URL}/models"
    try:
        r = requests.get(url, headers=_headers(), timeout=10)
        if r.status_code == 200:
            return True, f"OK {r.status_code} on {url}"
        return False, f"HTTP {r.status_code} on {url}: {r.text[:200]}"
    except Exception as e:
        return False, f"Request failed to {url}: {e}"

def _citation_map(passages: List[Any]) -> Dict[str, str]:
    """Map 'index' (as string) -> URL."""
    cmap: Dict[str, str] = {}
    for i, h in enumerate(passages, 1):
        p = _payload(h)
        url = p.get("url") or p.get("source_url") or ""
        if url:
            cmap[str(i)] = url
    return cmap

def linkify_citations(text: str, passages: List[Any]) -> str:
    """
    Turn [n] or [n, m] into clickable markdown links based on passages order.
    We escape the inner brackets like \[n] so Streamlit renders link text literally.
    """
    cmap = _citation_map(passages)

    # First: lists like [1, 3]
    def repl_list(m: re.Match) -> str:
        inner = m.group(1)  # "1, 3"
        out_parts: List[str] = []
        for token in re.split(r"\s*,\s*", inner):
            if token.isdigit() and token in cmap:
                out_parts.append(f"[\\[{token}\\]]({cmap[token]})")
            else:
                out_parts.append(token)
        return "[" + ", ".join(out_parts) + "]"

    text = re.sub(r"\[((?:\d+\s*,\s*)+\d+)\]", repl_list, text)

    # Then: singletons like [2]
    def repl_single(m: re.Match) -> str:
        n = m.group(1)
        url = cmap.get(n)
        return f"[\\[{n}\\]]({url})" if url else m.group(0)

    text = re.sub(r"\[(\d+)\]", repl_single, text)
    return text

def build_context(passages: List[Any], per_passage_chars: int = 900) -> str:
    """Build context string from passages with proper indexing."""
    blocks: List[str] = []
    for i, h in enumerate(passages, 1):
        p = _payload(h)
        title = (p.get("title") or "").strip()
        url   = (p.get("url") or p.get("source_url") or "").strip()
        body  = p.get("text") or p.get("chunk_text") or ""
        body  = body[:per_passage_chars]
        blocks.append(f"[{i}] {title}\nURL: {url}\n{body}")
    return "\n\n".join(blocks)

SYSTEM_PROMPT = """Tu es un assistant RH chargé de répondre à des questions dans le domaine des ressources humaines en t'appuyant sur les sources fournies.\n
        La date d'aujourd'hui est : {today} (au format AAAA-MM-JJ).\n\n
        ⚠️ Consigne temporelle : Les textes sources peuvent avoir été rédigés avant aujourd'hui 
        et mentionner des changements à venir. Interprète ces formulations en fonction de la date actuelle. 
        Si une mesure annoncée est déjà en vigueur aujourd'hui, écris-la au présent ou au passé, 
        jamais au futur.\n\n"
        
        Consignes :\n
        - Réponds de manière factuelle, concise et polie (vouvoiement).\n
        - Quand tu affirmes un fait, cite tes sources en fin de phrase avec le format [1], [2]… en te basant sur l'index de ces sources (ex: [1] est la source 1, [2] est la source 2, etc.)\n\n
        - Si l'information n'est pas présente dans les sources, réponds : \"Je suis navré, je n'ai pas trouvé la réponse à cette question\".\n\n
        - Si la question est mal formulée, réponds : \"Je ne comprends pas la question. Pourriez-vous reformuler ?\"\n\n
        - Ne fabrique pas de liens ni de références.\n\n
"""

def answer(query: str, passages: List[Any], stream: bool = False) -> str:
    """
    Generate an answer using LLM with proper citations.
    
    Args:
        query: User's question
        passages: List of retrieved documents for context
        stream: Whether to use streaming response (for real-time display)
        
    Returns:
        LLM response with clickable citations, or error message if failed
    """
    api_key = _get_api_key()

    # If targeting OpenAI endpoint explicitly, enforce API key presence
    if LLM_BASE_URL.startswith("https://api.openai.com") and not api_key:
        return ("❌ Aucune clé API trouvée. Définis LLM_API_KEY=sk-..."
                " dans ton .env")

    context = build_context(passages)
    system_prompt = SYSTEM_PROMPT.format(today=today_fr())

    user_prompt = textwrap.dedent(f"""\
    Question: {query}

    Sources (indexées) :
    {context}

    Rédigez la réponse aujourd'hui en citant les indices [n] correspondants.
    """)

    payload = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    url = f"{LLM_BASE_URL}/chat/completions"
    try:
        resp = requests.post(url, headers=_headers(), json=payload, timeout=90, stream=stream)
        if resp.status_code == 401:
            return (f"❌ 401 Unauthorized. Vérifie LLM_API_KEY et que "
                    f"{LLM_BASE_URL} attend bien 'Authorization: Bearer <clé>'. "
                    f"Détail: {resp.text}")
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"❌ Appel LLM échoué: {e}"

    if stream:
        chunks: List[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
                delta = j["choices"][0]["delta"].get("content", "")
            except Exception:
                delta = ""
            if delta:
                chunks.append(delta)
        return "".join(chunks)

    j = resp.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return f"❌ Réponse inattendue du LLM ({LLM_BASE_URL}): {j}"