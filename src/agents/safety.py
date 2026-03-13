from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    rate_limiter=rate_limiter
)
DISCLAIMER_CAREERS = (
    "⚠️ Esta resposta é baseada em vagas públicas do LinkedIn (Kaggle dataset) "
    "e pode não refletir o mercado atual. Não constitui aconselhamento profissional. "
    "Consulte recrutadores e profissionais da área para orientações personalizadas."
)

BLOCKED_PATTERNS = [
    "salário exato", "garantia de emprego", "processo seletivo específico",
    "dados pessoais", "informações confidenciais"
]

def safety_node(state: dict) -> dict:
    answer = state.get("answer", "")
    query = state.get("query", "").lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in query:
            return {
                "answer": (
                    f"Não posso fornecer informações sobre '{pattern}'. "
                    "Este sistema trabalha apenas com dados públicos de vagas. "
                    + DISCLAIMER_CAREERS
                ),
                "safety_disclaimer": DISCLAIMER_CAREERS
            }

    try:
        check_prompt = f"""A resposta abaixo contém algum conselho potencialmente prejudicial,
        como garantias de emprego, dados pessoais ou afirmações absolutas sobre salários?
        Responda apenas SIM ou NÃO.
        
        Resposta: {answer[:500]}"""

        risk = llm.invoke(check_prompt).content.strip().upper()
    except Exception:
        risk = "NÃO"

    if "SIM" in risk:
        answer = (
            "Posso fornecer informações gerais sobre o mercado de trabalho, "
            "mas não faço garantias ou promessas sobre resultados individuais. "
            + answer
        )

    return {
        "answer": answer,
        "safety_disclaimer": DISCLAIMER_CAREERS
    }
