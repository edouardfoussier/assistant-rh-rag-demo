# 🔎 RAG-RH Demo

A minimal **Retrieval-Augmented Generation (RAG)** assistant for HR-related questions, built as a portfolio project.  
It demonstrates the core components of a RAG pipeline: retrieval, reranking, and answer generation with a Large Language Model (LLM).

📊 Parameters you can tune (in the sidebar):
- **Top-K**: how many passages to keep after reranking  
- **Candidate-K**: how many candidates to fetch initially  
- **Reranker**: toggle on/off for higher accuracy vs. speed  

[Insert screenshot of the UI here]

---

## 🚀 Features

- **Data sources**: built on top of [AgentPublic’s Mediatech](https://huggingface.co/datasets/AgentPublic/Mediatech), with curated subsets I shared on Hugging Face 🤗 :
    - [`travail-emploi-clean`](https://huggingface.co/datasets/edouardfoussier/travail-emploi-clean)  
    - [`service-public-filtered`](https://huggingface.co/datasets/edouardfoussier/service-public-filtered)  
- **Vector store**: [Qdrant](https://qdrant.tech) for storing & searching embeddings.  
- **Retriever + Reranker**:  
  - Retriever = fast semantic search using embeddings.  
  - (Optional) Reranker = cross-encoder that reorders candidates for higher precision.  
- **LLM synthesis**: answers generated from retrieved passages, with clickable citations.  
- **UI**: built with [Streamlit](https://streamlit.io).  

---

## 🛠️ Tech Stack

- **Backend**: Python, Streamlit for UI  
- **Vector DB**: Qdrant  
- **Models**: Sentence-Transformers for embeddings, optional reranker, LLM via API  
- **Infra**: Docker & docker-compose  

---

## ⚙️ How it works (under the hood)

1. **Chunking & Embedding**  
   - Articles are split into small “chunks” of text.  
   - Each chunk is turned into a vector (numeric representation of its meaning).  
   - Stored in a **vector database (Qdrant)**.  

2. **Retrieval & Reranking**  
   - When you ask a question, it’s also turned into a vector.  
   - The **retriever** finds the most semantically similar chunks.  
   - (Optional) A **reranker** reorders them for higher precision.  

3. **Answer Generation**  
   - The top chunks are passed to a **Large Language Model (LLM)**.  
   - The LLM generates an answer, grounding it in the retrieved passages.  
   - Sources are cited inline with clickable references.  

---

## ⚡ Quickstart

```bash
# 0) Configure environment variables
cp .env.example .env

# Edit .env and set your LLM_API_KEY (e.g. OpenAI key)
# Optionally adjust LLM_MODEL / LLM_BASE_URL.

# 1) Start services
make up

# 2) Download cleaned datasets (from my HF repos)
make data

# 3) Ingest into Qdrant
make seed

# 4) Sanity check counts
make count

# You should see a total of 16,283 points

# 5) Open the UI
open http://localhost:8501

# (Windows: start http://localhost:8501)