from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
import logging
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_PATH = os.path.join(BASE_DIR, "logs", "mcp_calls.log")
FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss_index")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("mcp_jobs")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_PATH, mode="a")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

ALLOWED_TOOLS = ["search_jobs", "get_job_requirements"]

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings

app = Server("mcp-jobs")

@app.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="search_jobs",
            description="Busca vagas por cargo/área no corpus do LinkedIn",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_job_requirements",
            description="Retorna requisitos consolidados para um cargo específico",
            inputSchema={
                "type": "object",
                "properties": {"role": {"type": "string"}},
                "required": ["role"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name not in ALLOWED_TOOLS:
        logger.warning(f"MCP BLOCKED: tool={name} não está na allowlist")
        for h in logger.handlers:
            h.flush()
        raise ValueError(f"Tool '{name}' não permitida.")

    logger.info(f"MCP CALL: tool={name} args={arguments}")
    for h in logger.handlers:
        h.flush()

    vs = FAISS.load_local(
        FAISS_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

    if name == "search_jobs":
        docs = vs.similarity_search(arguments["query"], k=arguments.get("k", 5))
        results = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        return [types.TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]

    elif name == "get_job_requirements":
        docs = vs.similarity_search(f"requisitos {arguments['role']}", k=8)
        return [types.TextContent(type="text", text="\n---\n".join(d.page_content for d in docs))]

async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
