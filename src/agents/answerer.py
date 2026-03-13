from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from typing import List
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    rate_limiter=rate_limiter
)
def _format_citations(docs: List[Document]) -> list:
    citations = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        citations.append({
            "ref": i,
            "company": meta.get("company", "N/A"),
            "title": meta.get("title", "N/A"),
            "source": meta.get("source", "linkedin-job-postings-kaggle"),
            "snippet": doc.page_content[:200].strip()
        })
    return citations

def answerer_node(state: dict) -> dict:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {
            "answer": "Não encontrei informações suficientes para responder com segurança.",
            "citations": []
        }

    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        context_parts.append(
            f"[{i}] Cargo: {meta.get('title', '?')} | "
            f"Empresa: {meta.get('company', '?')}\n"
            f"{doc.page_content[:400]}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""Você é um assistente especialista em carreiras e mercado de trabalho.
Responda a pergunta do usuário com base EXCLUSIVAMENTE nos documentos abaixo.
Cite as fontes usando [número] inline na resposta.
Se a informação não estiver nos documentos, diga explicitamente.

Documentos:
{context}

Pergunta: {query}

Resposta (com citações obrigatórias no formato [1], [2], etc.):"""
    answer = llm.invoke(prompt).content
    citations = _format_citations(docs)
    return {"answer": answer, "citations": citations}
