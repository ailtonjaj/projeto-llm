from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import json
import re

rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    rate_limiter=rate_limiter
)

async def skill_gap_node(state: dict) -> dict:
    user_skills = state.get("user_skills", [])
    target_role = state.get("target_role") or state.get("query", "")

    server_params = StdioServerParameters(
        command="python",
        args=["src/mcp/server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            # Chama a ferramenta search_jobs diretamente
            search_tool = next((t for t in tools if t.name == "search_jobs"), None)
            if search_tool:
                docs_raw = await search_tool.ainvoke({"query": f"requisitos para {target_role}", "k": 8})
            else:
                docs_raw = "[]"

    prompt = f"""Com base nas vagas abaixo, identifique as skills mais exigidas para o cargo "{target_role}".
    Compare com as skills atuais do usuário e liste o GAP.

    Vagas recuperadas:
    {docs_raw}

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

    return {"skill_gap": gap_analysis, "retrieved_docs": []}
