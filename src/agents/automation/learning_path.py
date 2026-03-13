from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    rate_limiter=rate_limiter
)

def learning_path_node(state: dict) -> dict:
    gap = state.get("skill_gap", {})
    target_role = state.get("target_role", "")

    prompt = f"""Crie uma trilha de estudos priorizada para alguém que quer ser "{target_role}".
    Skills que precisa aprender: {gap.get('gap', [])}
    
    Retorne JSON:
    {{
      "trilha": [
        {{
          "fase": 1,
          "titulo": "...",
          "skills": ["..."],
          "recursos": ["curso/livro/doc sugerido"],
          "duracao_estimada": "X semanas",
          "prioridade": "alta/media/baixa"
        }}
      ],
      "tempo_total_estimado": "X meses",
      "disclaimer": "Esta trilha é baseada em vagas públicas do LinkedIn. Resultados podem variar."
    }}
    Retorne APENAS o JSON, sem texto antes ou depois.
    """

    raw = llm.invoke(prompt).content

    try:
        # Tenta direto primeiro
        learning_path = json.loads(raw)
    except json.JSONDecodeError:
        try:
            # Extrai o primeiro bloco JSON encontrado no texto
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                learning_path = json.loads(match.group())
            else:
                raise ValueError("Nenhum JSON encontrado")
        except Exception:
            learning_path = {
                "trilha": [],
                "tempo_total_estimado": "indefinido",
                "disclaimer": "Não foi possível gerar a trilha. Tente novamente."
            }

    return {"learning_path": learning_path}
