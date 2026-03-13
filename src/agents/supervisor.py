from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    rate_limiter=rate_limiter
)
def supervisor_node(state: dict) -> dict:
    prompt = f"""Classifique a intenção do usuário:
    Pergunta: {state.get('query', '')}
    Skills informadas: {state.get('user_skills', [])}
    Cargo alvo: {state.get('target_role', '')}
    
    Responda APENAS com: "qa", "automation" ou "refuse"
    - qa: pergunta sobre vagas, requisitos, mercado
    - automation: quer trilha de estudos / gap de skills
    - refuse: fora de escopo ou potencialmente prejudicial
    """
    intent = llm.invoke(prompt).content.strip().lower()
    if intent not in ["qa", "automation", "refuse"]:
        intent = "qa"
    return {"intent": intent}
