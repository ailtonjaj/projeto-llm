import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ragas import evaluate, RunConfig, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from ragas.dataset_schema import SingleTurnSample
from src.graph.graph import build_graph
import pandas as pd

TEST_QUESTIONS = [
    {"question": "Quais skills são exigidas para Data Scientist?",
     "ground_truth": "Python, machine learning, SQL e estatística são as skills mais comuns para Data Scientist."},
    {"question": "O que um Backend Developer precisa saber?",
     "ground_truth": "Backend developers precisam de Python ou Java, APIs REST, bancos de dados e Docker."},
    {"question": "Quais são os requisitos para Product Manager?",
     "ground_truth": "Product managers precisam de habilidades em Agile, roadmap, comunicação e análise de dados."},
    {"question": "O que é exigido para DevOps Engineer?",
     "ground_truth": "DevOps engineers precisam de Kubernetes, CI/CD, Terraform, Linux e cloud computing."},
    {"question": "Quais skills um UX Designer precisa ter?",
     "ground_truth": "UX Designers precisam de Figma, pesquisa com usuários, prototipação e design thinking."},
    {"question": "O que um Frontend Developer precisa saber?",
     "ground_truth": "Frontend developers precisam de JavaScript, React ou Vue, HTML, CSS e responsividade."},
    {"question": "Quais são os requisitos para Machine Learning Engineer?",
     "ground_truth": "ML Engineers precisam de Python, TensorFlow ou PyTorch, matemática e cloud computing."},
    {"question": "O que um Data Engineer precisa dominar?",
     "ground_truth": "Data Engineers precisam de SQL, Python, pipelines de dados, Spark e ferramentas de cloud."},
    {"question": "Quais habilidades são necessárias para Cloud Architect?",
     "ground_truth": "Cloud Architects precisam de AWS ou Azure ou GCP, redes, segurança e infraestrutura como código."},
    {"question": "O que um Cybersecurity Analyst precisa saber?",
     "ground_truth": "Analistas de segurança precisam de redes, criptografia, análise de vulnerabilidades e SIEM."},
    {"question": "Quais skills são comuns em vagas de Full Stack Developer?",
     "ground_truth": "Full Stack Developers precisam de JavaScript, Node.js, React, bancos de dados e APIs REST."},
    {"question": "O que um Mobile Developer precisa dominar?",
     "ground_truth": "Mobile developers precisam de Swift ou Kotlin, React Native ou Flutter e publicação em lojas."},
    {"question": "Quais são os requisitos para Business Analyst?",
     "ground_truth": "Business Analysts precisam de SQL, Excel, requisitos de negócio, Power BI e comunicação."},
    {"question": "O que um Scrum Master precisa saber?",
     "ground_truth": "Scrum Masters precisam de metodologias ágeis, Scrum, facilitação e ferramentas como Jira."},
    {"question": "Quais habilidades são exigidas para Data Analyst?",
     "ground_truth": "Data Analysts precisam de SQL, Excel, Python ou R, visualização de dados e estatística básica."},
]

def run_eval():
    rate_limiter = InMemoryRateLimiter(requests_per_second=0.3)

    ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        rate_limiter=rate_limiter
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    run_config = RunConfig(
        max_retries=5,
        max_wait=120,
        timeout=300
    )

    graph = build_graph()
    all_results = []
    latencias = []

    for item in TEST_QUESTIONS:
        print(f"\nAvaliando: {item['question'][:50]}...")
        start = time.time()

        try:
            result = graph.invoke({
                "query": item["question"],
                "retry_count": 0,
                "user_skills": [],
                "target_role": "",
            })
            elapsed = time.time() - start
            latencias.append(elapsed)
            print(f"  RAG OK ({elapsed:.1f}s) — avaliando métricas...")

            sample = SingleTurnSample(
                user_input=item["question"],
                response=result.get("answer", ""),
                retrieved_contexts=[d.page_content for d in result.get("retrieved_docs", [])],
                reference=item["ground_truth"],
            )

            dataset = EvaluationDataset(samples=[sample])
            time.sleep(3)

            row_result = evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=run_config
            )

            row_df = row_result.to_pandas()
            row_df["question"] = item["question"]
            all_results.append(row_df)
            print(f"  Métricas: {row_result}")

        except Exception as e:
            print(f"  ERRO: {e}")
            latencias.append(time.time() - start)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv("eval/ragas_results.csv", index=False)

        print("\n=== Resultados RAGAS ===")
        numeric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        existing_cols = [c for c in numeric_cols if c in final_df.columns]
        print(final_df[existing_cols].describe())
        print(f"\nMédias finais:")
        print(final_df[existing_cols].mean())
        print(f"\nLatência média: {sum(latencias)/len(latencias):.1f}s")
        print(f"Resultados salvos em eval/ragas_results.csv")
    else:
        print("Nenhum resultado calculado.")

if __name__ == "__main__":
    run_eval()
