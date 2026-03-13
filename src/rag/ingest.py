import kagglehub
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.vectorstore import build_vectorstore

def load_kaggle_docs(max_rows: int = 5000) -> list[Document]:
    print("Baixando dataset do Kaggle...")
    path = kagglehub.dataset_download("arshkon/linkedin-job-postings")

    jobs_df = pd.read_csv(f"{path}/job_postings.csv", nrows=max_rows)
    jobs_df = jobs_df.dropna(subset=["description"])

    # Tenta enriquecer com skills se existir arquivo separado
    try:
        skills_df = pd.read_csv(f"{path}/job_skills.csv")
        jobs_df = jobs_df.merge(skills_df, on="job_id", how="left")
    except Exception:
        jobs_df["skill_abr"] = ""

    docs = []
    for _, row in jobs_df.iterrows():
        text = "\n".join(filter(None, [
            f"Cargo: {row.get('title', '')}",
            f"Empresa: {row.get('company_name', '')}",
            f"Local: {row.get('location', '')}",
            f"Nível: {row.get('formatted_experience_level', '')}",
            f"Tipo: {row.get('formatted_work_type', '')}",
            f"Skills: {row.get('skill_abr', '')}",
            f"Descrição: {str(row.get('description', ''))[:800]}",
        ]))

        docs.append(Document(
            page_content=text,
            metadata={
                "job_id": str(row.get("job_id", "")),
                "title": str(row.get("title", "")),
                "company": str(row.get("company_name", "")),
                "location": str(row.get("location", "")),
                "experience_level": str(row.get("formatted_experience_level", "")),
                "source": "linkedin-job-postings-kaggle",
            }
        ))

    print(f"Carregados {len(docs)} documentos.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Gerados {len(chunks)} chunks. Indexando no FAISS...")

    build_vectorstore(chunks)
    print("Indexação concluída.")
    return chunks


if __name__ == "__main__":
    load_kaggle_docs()
