from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    rate_limiter=rate_limiter
)
def self_check_node(state: dict) -> dict:
    answer = state.get("answer", "")
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {"self_check_passed": False}

    docs_text = "\n---\n".join([d.page_content for d in retrieved_docs])

    prompt = f"""Verifique se a resposta abaixo está TOTALMENTE suportada pelos documentos.
    Documentos recuperados:
    {docs_text}
    
    Resposta gerada:
    {answer}
    
    Responda APENAS "SIM" se toda afirmação tem suporte, ou "NÃO" se há afirmações sem evidência.
    """
    result = llm.invoke(prompt).content.strip().upper()
    return {"self_check_passed": result == "SIM"}
