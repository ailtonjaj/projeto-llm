import kagglehub
import pandas as pd
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_ingest():
    if os.path.exists("data/faiss_index/index.faiss"):
        print("Índice FAISS já existe — pulando ingestão.")
        return

    path = kagglehub.dataset_download("arshkon/linkedin-job-postings")
    print(f"Dataset em: {path}")
    postings_path = str(Path(path) / "postings.csv")
    jobs_df = pd.read_csv(
        postings_path,
        usecols=["job_id", "title", "description", "skills_desc"],
        nrows=5000
    )
    jobs_df = jobs_df.dropna(subset=["description"])
    print(f"Total de vagas carregadas: {len(jobs_df)}")
    docs = []
    for _, row in jobs_df.iterrows():
        text = "\n".join(filter(None, [
            f"Cargo: {row.get('title', '')}",
            f"Skills: {row.get('skills_desc', '')}",
            f"Descrição: {str(row.get('description', ''))[:800]}",
        ]))
        docs.append(Document(
            page_content=text,
            metadata={
                "title":  str(row.get("title", "")),
                "job_id": str(row.get("job_id", "")),
                "source": "linkedin-job-postings-kaggle",
            }
        ))
    print(f"Documentos criados: {len(docs)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Chunks gerados: {len(chunks)} — iniciando indexação FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs("data", exist_ok=True)
    vectorstore.save_local("data/faiss_index")
    print(f"Indexação concluída: {len(chunks)} chunks de {len(docs)} vagas.")

if __name__ == "__main__":
    load_and_ingest()
