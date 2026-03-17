from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import os

FAISS_PATH = "data/faiss_index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_embeddings = None
_vectorstore = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(
                f"Índice FAISS não encontrado em '{FAISS_PATH}'. "
                "Execute 'python ingest/ingest_kaggle.py' primeiro."
            )
        _vectorstore = FAISS.load_local(
            FAISS_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    return _vectorstore

def build_vectorstore(docs: List[Document]) -> FAISS:
    vs = FAISS.from_documents(docs, get_embeddings())
    os.makedirs("data", exist_ok=True)
    vs.save_local(FAISS_PATH)
    global _vectorstore
    _vectorstore = vs
    return vs

def similarity_search(query: str, k: int = 5) -> List[Document]:
    return get_vectorstore().similarity_search(query, k=k)
