"""
Ingestion de parquet avec embeddings existants -> Qdrant.
"""

import argparse
import ast
import math
from typing import Any, List
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams



def parse_vec(val: Any) -> List[float]:
    """Convertit une cellule 'embeddings' en list[float]."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return list(ast.literal_eval(val))
    raise ValueError("Embedding must be list[float] or string")


def check_norms(vectors: List[List[float]], sample_size: int = 1000) -> None:
    """Sanity check sur la norme L2 des embeddings."""
    if not vectors:
        print("⚠️ Pas de vecteurs à vérifier")
        return
    sample = vectors[: min(sample_size, len(vectors))]
    norms = [math.sqrt(sum(x * x for x in v)) for v in sample]
    mean_norm = float(np.mean(norms))
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    print(
        f"🧪 Embedding norms (sample {len(sample)}): "
        f"mean={mean_norm:.4f}, min={min_norm:.4f}, max={max_norm:.4f}"
    )
    if abs(mean_norm - 1.0) > 0.05:
        print("⚠️ Warning: embeddings do not look normalized (mean norm ≠ 1.0)")
    else:
        print("✅ Embeddings look normalized")


def main():
    ap = argparse.ArgumentParser("Ingest existing-embeddings parquet -> Qdrant")
    ap.add_argument("--parquet", required=True, help="Chemin vers le fichier .parquet")
    ap.add_argument("--collection", default="rag_rh_chunks")
    ap.add_argument("--qdrant-url", default="http://qdrant:6333")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="Si présent, recrée la collection (⚠️ supprime les points existants).",
    )
    ap.add_argument(
        "--embed-col",
        default="embeddings_bge-m3",
        help="Nom de la colonne contenant les embeddings.",
    )
    args = ap.parse_args()

    print(f"📦 Lecture du parquet : {args.parquet}")
    df = pd.read_parquet(args.parquet)

    if args.embed_col not in df.columns:
        raise SystemExit(f"Colonne embeddings introuvable: {args.embed_col}")

    # Dimensions à partir du 1er embedding
    first = parse_vec(df[args.embed_col].iloc[0])
    dims = len(first)
    print(f"ℹ️ Dimension embeddings détectée : {dims}")

    # Sanity check des normes
    vectors_list = [parse_vec(v) for v in df[args.embed_col].tolist()]
    check_norms(vectors_list)

    # Connexion Qdrant
    qc = QdrantClient(url=args.qdrant_url)

    # Création / recréation collection
    if args.recreate:
        print(f"🆕 Recreate collection: {args.collection}")
        qc.recreate_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )
    else:
        if not qc.collection_exists(args.collection):
            print(f"🆕 Création de la collection {args.collection}")
            qc.create_collection(
                collection_name=args.collection,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )
        else:
            print(f"ℹ️ Collection exists: {args.collection} (incremental upsert)")

    # Upsert par batch
    points = []
    total = 0
    for i, row in df.iterrows():
        vec = parse_vec(row[args.embed_col])
        pid = str(row.get("chunk_id", i))

        payload = {
            "sid": row.get("sid", ""),
            "chunk_index": row.get("chunk_index", ""),
            "title": row.get("title", ""),
            "surtitle": row.get("surtitle", ""),
            "source": row.get("source", ""),
            "introduction": row.get("introduction", ""),
            "date": row.get("date", ""),
            "url": row.get("url", ""),
            "context": row.get("context", ""),
            "text": row.get("text", ""),
            "theme": row.get("theme", ""),
            "audience": row.get("audience", ""),
            "updated_at": row.get("date", ""),
            "related_questions": row.get("related_questions", ""),
            "web_services": row.get("web_services", ""),
        }

        points.append(PointStruct(id=pid, vector=vec, payload=payload))

        if len(points) >= args.batch:
            qc.upsert(collection_name=args.collection, points=points)
            total += len(points)
            print(f"⬆️ Upsert {total} points")
            points = []

    if points:
        qc.upsert(collection_name=args.collection, points=points)
        total += len(points)

    print(f"✅ Ingestion terminée: {total} points")


if __name__ == "__main__":
    main()