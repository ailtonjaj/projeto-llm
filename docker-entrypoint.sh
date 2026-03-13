#!/bin/bash
set -e

echo "Verificando índice FAISS..."
if [ ! -d "data/faiss_index" ]; then
    echo "Índice não encontrado — indexando..."
    python ingest/ingest_kaggle.py
else
    echo "Índice FAISS já existe — pulando ingestão."
fi

echo "Iniciando aplicação Streamlit..."
PYTHONPATH=/app streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0
