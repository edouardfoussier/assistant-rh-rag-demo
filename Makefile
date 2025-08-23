up:
\tdocker compose up -d

down:
\tdocker compose down

rebuild:
\tdocker compose build --no-cache

logs:
\tdocker compose logs -f app

seed:
\tdocker compose exec app python /app/scripts/seed_qdrant.py \
\t  --parquet /app/rag_app/data/service-public.clean.parquet \
\t  --qdrant-url http://qdrant:6333 --collection rag_rh_chunks --recreate

fmt:
\trufflehog --version >/dev/null 2>&1 || true
\tpython -m black app/ scripts/