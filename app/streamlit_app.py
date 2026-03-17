import streamlit as st
from src.graph.graph import build_graph

st.set_page_config(page_title="Career Guide Agent", page_icon="🎯")
st.title("🎯 Career Guide Agent")
st.caption("Descubra o que estudar para conquistar a vaga dos seus sonhos")

graph = build_graph()

tab1, tab2 = st.tabs(["🔍 Pesquisar Vagas", "🗺️ Trilha de Estudos"])

with tab1:
    query = st.text_input("Pergunta sobre vagas ou mercado:")
    if st.button("Buscar") and query:
        with st.spinner("Consultando..."):
            result = graph.invoke({"query": query, "retry_count": 0})
        st.markdown(result.get("answer", "Sem resposta."))
        if result.get("citations"):
            with st.expander("📚 Fontes citadas"):
                for c in result["citations"]:
                    st.write(f"- {c}")
        if result.get("safety_disclaimer"):
            st.info(result["safety_disclaimer"])

with tab2:
    target_role = st.text_input("Cargo que você quer alcançar:", placeholder="Ex: Data Scientist")
    skills_input = st.text_area("Suas skills atuais (uma por linha):", placeholder="Python\nSQL\nExcel")
    
    if st.button("Gerar Trilha") and target_role:
        user_skills = [s.strip() for s in skills_input.split("\n") if s.strip()]
        with st.spinner("Analisando vagas e gerando sua trilha personalizada..."):
            result = graph.invoke({
                "query": f"trilha para {target_role}",
                "intent": "automation",
                "user_skills": user_skills,
                "target_role": target_role,
                "retry_count": 0
            })
        
        lp = result.get("learning_path", {})
        gap = result.get("skill_gap", {})
        
        if gap.get("gap"):
            st.subheader("📊 Skills que você precisa desenvolver:")
            cols = st.columns(3)
            for i, skill in enumerate(gap["gap"]):
                cols[i % 3].warning(f"• {skill}")
        
        if lp.get("trilha"):
            st.subheader(f"🗺️ Sua Trilha — {lp.get('tempo_total_estimado', '')}")
            for fase in lp["trilha"]:
                with st.expander(f"Fase {fase['fase']}: {fase['titulo']} ({fase['duracao_estimada']})"):
                    st.write("**Skills:**", ", ".join(fase.get("skills", [])))
                    st.write("**Recursos sugeridos:**")
                    for r in fase.get("recursos", []):
                        st.write(f"  - {r}")
        
        if lp.get("disclaimer"):
            st.caption(f"⚠️ {lp['disclaimer']}")
