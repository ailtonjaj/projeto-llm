from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 15 perguntas rotuladas sobre vagas
test_questions = [
    {"question": "Quais skills são exigidas para Data Scientist?",
     "ground_truth": "Python, ML, SQL, estatística são as mais comuns"},
    # ... +14 questões
]

def run_eval(graph):
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in test_questions:
        result = graph.invoke({"query": item["question"], "retry_count": 0})
        data["question"].append(item["question"])
        data["answer"].append(result.get("answer", ""))
        data["contexts"].append([d.page_content for d in result.get("retrieved_docs", [])])
        data["ground_truth"].append(item["ground_truth"])
    
    dataset = Dataset.from_dict(data)
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                         context_precision, context_recall])
    print(results)
