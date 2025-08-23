# Makefile for RAG-RH Demo

DATA_DIR=app/rag_app/data
QDRANT_URL=http://qdrant:6333
COLLECTION=rag_rh_chunks

.PHONY: all data seed count up down logs

all: up data seed count

# -----------------
# Infra
# -----------------
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f app

# -----------------
# Data
# -----------------
data:
	mkdir -p $(DATA_DIR)
	curl -L -o $(DATA_DIR)/service-public-filtered.parquet \
	  https://huggingface.co/datasets/edouardfoussier/service-public-filtered/resolve/main/service-public-filtered.parquet
	curl -L -o $(DATA_DIR)/travail-emploi-clean.parquet \
	  https://huggingface.co/datasets/edouardfoussier/travail-emploi-clean/resolve/main/travail-emploi-clean.parquet

# -----------------
# Seeding Qdrant
# -----------------
seed:
	docker exec rag-app python /app/scripts/seed_qdrant.py \
	  --parquet /app/rag_app/data/travail-emploi-clean.parquet \
	  --qdrant-url $(QDRANT_URL) --collection $(COLLECTION)
	docker exec rag-app python /app/scripts/seed_qdrant.py \
	  --parquet /app/rag_app/data/service-public-filtered.parquet \
	  --qdrant-url $(QDRANT_URL) --collection $(COLLECTION)

count:
	@docker compose exec -T app python -c 'from qdrant_client import QdrantClient; qc = QdrantClient(url="$(QDRANT_URL)"); print(qc.count("$(COLLECTION)", exact=True))'