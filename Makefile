# -------- config --------
APP_SVC ?= app
QDRANT_SVC ?= qdrant
COMPOSE ?= docker compose

DATA_DIR := app/rag_app/data
SP_FILE  := $(DATA_DIR)/service-public-filtered.parquet
TE_FILE  := $(DATA_DIR)/travail-emploi-clean.parquet

SP_URL   := https://huggingface.co/datasets/edouardfoussier/service-public-filtered/resolve/main/service-public-filtered.parquet
TE_URL   := https://huggingface.co/datasets/edouardfoussier/travail-emploi-clean/resolve/main/travail-emploi-clean.parquet

COLLECTION := rag_rh_chunks
QDRANT_URL := http://qdrant:6333

# -------- phony --------
.PHONY: up down rebuild logs data seed seed-sp seed-te count clean nuke

# -------- lifecycle --------
up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

rebuild:
	$(COMPOSE) build $(APP_SVC) && $(COMPOSE) up -d $(APP_SVC)

logs:
	$(COMPOSE) logs -f $(APP_SVC)

# -------- data --------
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(SP_FILE): | $(DATA_DIR)
	curl -L -o $(SP_FILE) $(SP_URL)

$(TE_FILE): | $(DATA_DIR)
	curl -L -o $(TE_FILE) $(TE_URL)

data: $(SP_FILE) $(TE_FILE)
	@echo "✅ datasets downloaded into $(DATA_DIR)"

# -------- ingestion --------
seed: seed-te seed-sp

seed-te:
	$(COMPOSE) exec -T $(APP_SVC) \
	  python /app/scripts/seed_qdrant.py \
	    --parquet /app/rag_app/data/travail-emploi-clean.parquet \
	    --qdrant-url $(QDRANT_URL) \
	    --collection $(COLLECTION)

seed-sp:
	$(COMPOSE) exec -T $(APP_SVC) \
	  python /app/scripts/seed_qdrant.py \
	    --parquet /app/rag_app/data/service-public-filtered.parquet \
	    --qdrant-url $(QDRANT_URL) \
	    --collection $(COLLECTION)

count:
	$(COMPOSE) exec -T $(APP_SVC) python - <<'PY'
from qdrant_client import QdrantClient
qc = QdrantClient(url="$(QDRANT_URL)")
print(qc.count("$(COLLECTION)", exact=True))
PY

# -------- cleanup --------
clean:
	$(COMPOSE) down -v

# nuclear option: remove local images too
nuke:
	$(COMPOSE) down -v --rmi local