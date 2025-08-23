FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.6.0"

COPY app/requirements.txt /app/requirements.txt
RUN PIP_INDEX_URL= pip install --no-cache-dir -r /app/requirements.txt

COPY app /app
COPY scripts /app/scripts

CMD streamlit run rag_app/app_ui.py --server.address 0.0.0.0 --server.port ${STREAMLIT_SERVER_PORT:-8501}