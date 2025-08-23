# 🔎 RAG-RH Demo

A minimal **Retrieval-Augmented Generation (RAG)** assistant for HR-related questions, built as a portfolio project.  
It demonstrates the core components of a RAG pipeline: retrieval, reranking, and answer generation with a Large Language Model (LLM).

📊 Parameters you can tune

In the sidebar of the UI:
	•	Top-K: how many passages to keep after reranking
	•	Candidate-K: how many candidates to fetch initially
	•	Reranker: toggle on/off for higher accuracy vs. speed

👉 Live demo: [rag.edouardfoussier.com](https://rag.edouardfoussier.com)  

---

## 🚀 Features

- **Data sources**: public datasets from [AgentPublic’s Mediatech](https://huggingface.co/datasets/AgentPublic/Mediatech) (articles from *travail-emploi* and *service-public*).  
- **Vector store**: [Qdrant](https://qdrant.tech) for storing & searching embeddings.  
- **Retriever + Reranker**:  
  - Retriever = fast semantic search using embeddings.  
  - (Optional) Reranker = cross-encoder that reorders candidates for higher precision.  
- **LLM synthesis**: answers generated from retrieved passages, with clickable citations.  
- **UI**: built with [Streamlit](https://streamlit.io).  

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI (internal utilities), Streamlit for UI  
- **Vector DB**: Qdrant  
- **Models**: Sentence-Transformers for embeddings, optional reranker, LLM via API  
- **Infra**: Docker & docker-compose  

---

## ⚡ Quickstart

Clone and run with Docker:

```bash
git clone https://github.com/yourname/rag-rh-demo.git
cd rag-rh-demo

# Copy environment variables
cp .env.example .env

# Start services
docker compose up --build

```bash
# 1) Clone + env
cp .env.example .env
# Edit .env (choose your LLM: OpenAI or Ollama, see below)

# 2) Start services
docker compose up -d

# 3) Ingest data (parquet already contains embeddings)
docker compose exec app python /app/scripts/seed_qdrant.py \
  --parquet /app/rag_app/data/service-public.clean.parquet \
  --qdrant-url http://qdrant:6333 \
  --collection rag_rh_chunks \
  --recreate


  ⚠️ Disclaimer

This is a demo / MVP, not production-ready.
It provides public information only — not legal or HR advice.

⸻

👨‍💻 Built by Edouard Foussier as a RAG project portfolio.