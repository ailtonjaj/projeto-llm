# JobPath: Agente de Trilha de Carreira baseado em Vagas de Emprego

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Objetivo

O **JobPath** é uma prova de conceito (PoC) open source de um sistema agêntico que ajuda estudantes e profissionais a identificar **trilhas de conhecimento e habilidades** necessárias para diferentes profissões.

O sistema utiliza dados públicos de vagas de emprego (Kaggle) e, via agentes LangChain, **indexa requisitos, gera Q&A com citações**, e cria **trilhas de estudo personalizadas** para cada tipo de carreira.

O projeto tem apelo social ao **orientar pessoas em transição de carreira ou capacitação profissional**, facilitando acesso à informação de forma estruturada e confiável.

---

## Funcionalidades

1. **Indexação de Documentos**
   - Base de dados: conjuntos públicos de vagas de emprego do Kaggle.
   - Indexa os **requisitos das vagas** e cria embeddings para recuperação semântica.

2. **Sistema Agêntico (LangChain)**
   - **Supervisor Agent**: decide se a consulta é Q&A, automação ou recusa.
   - **Retriever Agent**: busca e organiza trechos relevantes das vagas.
   - **Safety/Policy Agent**: inclui disclaimers de orientação profissional e evita aconselhamento arriscado.
   - **Answerer/Writer Agent**: responde perguntas formatadas com citações (trechos de vagas e URLs/IDs).
   - **Self-Check Agent**: verifica se as respostas estão suportadas por evidências, re-busca ou recusa quando necessário.

3. **Automação**
   - **Automation Agent**: gera **trilhas de estudo personalizadas** com base nos requisitos extraídos.
   - Exemplo de fluxo:
     1. Seleciona vaga alvo.
     2. Extrai habilidades e requisitos.
     3. Gera plano de estudo com tópicos, cursos recomendados e ordem sugerida.
     4. Log de etapas e decisões do agente.

4. **MCP (Model Context Protocol)**
   - Integração com MCP local (`mcp-docstore`) que expõe a base de vagas como ferramenta padronizada.
   - Controla acesso via **allowlist** e registra chamadas de tool.
   - Permite ao agente consultar, listar e validar documentos sem risco de exfiltração.

5. **UI**
   - Interface interativa via **Streamlit**, permitindo:
     - Perguntar sobre requisitos de vagas.
     - Gerar trilhas de estudo personalizadas.
     - Visualizar citações e evidências.

---

## Stack Técnico

- **Linguagem:** Python 3.11+
- **Agentes:** LangChain + LangGraph
- **RAG / Indexação:** FAISS ou Chroma (open-source)
- **LLM:** Llama 3.x (via Ollama) — local
- **Embeddings:** HuggingFace (ex.: `bge-m3` ou `gte/bge-small`)
- **UI:** Streamlit
- **MCP:** MCP-docstore local
- **Código & Versionamento:** GitHub (MIT License)

---

## Arquitetura

```text
[User Query]
     ↓
[Supervisor Agent] → decide rota
     ├─> Q&A → [Retriever Agent] → [Self-Check] → [Answerer/Writer] → Resposta c/ citações
     └─> Automação → [Automation Agent] → [RAG lookup] → Trilha de estudo → Output
            ↑
            └─ MCP-docstore → dados de vagas indexados
```
