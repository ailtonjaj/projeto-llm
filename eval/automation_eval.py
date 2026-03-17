import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.graph import build_graph

AUTOMATION_TASKS = [
    {
        "id": 1,
        "target_role": "Data Scientist",
        "user_skills": ["Python", "Excel"],
        "expected_gap_contains": ["machine learning", "sql", "statistics"],
        "expected_phases_min": 2,
    },
    {
        "id": 2,
        "target_role": "Backend Developer",
        "user_skills": ["HTML", "CSS"],
        "expected_gap_contains": ["python", "api", "docker"],
        "expected_phases_min": 2,
    },
    {
        "id": 3,
        "target_role": "Product Manager",
        "user_skills": ["Excel", "PowerPoint"],
        "expected_gap_contains": ["agile", "roadmap"],
        "expected_phases_min": 1,
    },
    {
        "id": 4,
        "target_role": "DevOps Engineer",
        "user_skills": ["Linux", "Python"],
        "expected_gap_contains": ["kubernetes", "ci/cd", "terraform"],
        "expected_phases_min": 2,
    },
    {
        "id": 5,
        "target_role": "UX Designer",
        "user_skills": ["Photoshop"],
        "expected_gap_contains": ["figma", "user research"],
        "expected_phases_min": 1,
    },
]

def run_automation_eval():
    graph = build_graph()
    results = []

    for task in AUTOMATION_TASKS:
        print(f"\nTarefa {task['id']}: {task['target_role']}")
        start = time.time()

        try:
            state = graph.invoke({
                "query": f"trilha para {task['target_role']}",
                "intent": "automation",
                "user_skills": task["user_skills"],
                "target_role": task["target_role"],
                "retry_count": 0,
            })
            elapsed = time.time() - start

            gap = state.get("skill_gap", {}).get("gap", [])
            trilha = state.get("learning_path", {}).get("trilha", [])

            gap_lower = " ".join(gap).lower()
            gap_hits = sum(
                1 for s in task["expected_gap_contains"]
                if s.lower() in gap_lower
            )
            gap_precision = gap_hits / len(task["expected_gap_contains"])
            phases_ok = len(trilha) >= task["expected_phases_min"]
            success = gap_precision >= 0.5 and phases_ok

            result = {
                "task_id": task["id"],
                "role": task["target_role"],
                "success": success,
                "gap_precision": round(gap_precision, 2),
                "phases_generated": len(trilha),
                "time_seconds": round(elapsed, 2),
            }

        except Exception as e:
            result = {
                "task_id": task["id"],
                "role": task["target_role"],
                "success": False,
                "error": str(e),
                "time_seconds": round(time.time() - start, 2),
            }

        results.append(result)
        print(result)

    successes = [r for r in results if r.get("success")]
    times = [r["time_seconds"] for r in results]
    avg_precision = sum(r.get("gap_precision", 0) for r in results) / len(results)

    print("\n=== Resumo da Avaliação de Automação ===")
    print(f"Taxa de sucesso:     {len(successes)}/{len(results)} ({100*len(successes)//len(results)}%)")
    print(f"Tempo médio:         {sum(times)/len(times):.1f}s")
    print(f"Gap precision média: {avg_precision:.2f}")
    return results

if __name__ == "__main__":
    run_automation_eval()
