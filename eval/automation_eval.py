import time
from src.graph.workflow import build_graph

graph = build_graph()

# 5 tarefas de automação com input e output esperado
AUTOMATION_TASKS = [
    {
        "id": 1,
        "target_role": "Data Scientist",
        "user_skills": ["Python", "Excel"],
        "expected_gap_contains": ["Machine Learning", "SQL", "Statistics"],
        "expected_phases_min": 2,
    },
    {
        "id": 2,
        "target_role": "Backend Developer",
        "user_skills": ["HTML", "CSS"],
        "expected_gap_contains": ["Python", "API", "Docker"],
        "expected_phases_min": 2,
    },
    {
        "id": 3,
        "target_role": "Product Manager",
        "user_skills": ["Excel", "PowerPoint"],
        "expected_gap_contains": ["Agile", "Roadmap"],
        "expected_phases_min": 1,
    },
    {
        "id": 4,
        "target_role": "DevOps Engineer",
        "user_skills": ["Linux", "Python"],
        "expected_gap_contains": ["Kubernetes", "CI/CD", "Terraform"],
        "expected_phases_min": 2,
    },
    {
        "id": 5,
        "target_role": "UX Designer",
        "user_skills": ["Photoshop"],
        "expected_gap_contains": ["Figma", "User Research"],
        "expected_phases_min": 1,
    },
]

def run_automation_eval():
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

            # Checa se gap contém skills esperadas (case-insensitive)
            gap_lower = " ".join(gap).lower()
            gap_hits = sum(
                1 for s in task
