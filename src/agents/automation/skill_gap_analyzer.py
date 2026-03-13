from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
import re
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    rate_limiter=rate_limiter
)

def skill_gap_node(state: dict) -> dict:
    user_skills = state.get("user_skills", [])
    target_role = state.get("target_role") or state.get("query", "")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vs = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vs.similarity_search(f"requisitos para {target_role}", k=8)

    req_text = "\n".join([d.page_content for d in docs])

    prompt = f"""Com base nas vagas abaixo, identifique as skills mais exigidas para o cargo "{target_role}".
    Compare com as skills atuais do usuário e liste o GAP.

    Vagas recuperadas:
    {req_text}

    Skills atuais do usuário: {user_skills}

    Retorne JSON com:
    {{
      "required_skills": ["skill1", ...],
      "user_has": ["skill1", ...],
      "gap": ["skill_faltante1", ...],
      "sources": ["empresa/vaga citada", ...]
    }}
    Retorne APENAS o JSON, sem texto antes ou depois.
    """

    raw = llm.invoke(prompt).content

    try:
        gap_analysis = json.loads(raw)
    except json.JSONDecodeError:
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                gap_analysis = json.loads(match.group())
            else:
                raise ValueError("Nenhum JSON encontrado")
        except Exception:
            gap_analysis = {
                "required_skills": [],
                "user_has": user_skills,
                "gap": [],
                "sources": []
            }

    return {"skill_gap": gap_analysis, "retrieved_docs": docs}
