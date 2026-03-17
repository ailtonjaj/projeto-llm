from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings

def retriever_node(state: dict) -> dict:
    vs = FAISS.load_local(
        "data/faiss_index",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )
    query = state.get("target_role") or state.get("query", "")
    docs = vs.similarity_search(query, k=5)
    return {
        "retrieved_docs": docs,
        "retry_count": state.get("retry_count", 0) + 1
    }
