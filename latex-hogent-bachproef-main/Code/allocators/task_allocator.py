from ai.predictor import score_employee
import json

with open("models/role_map.json") as f:
    role_map = json.load(f)

def assign_task(task, employees):
    best_employee = None
    best_score = -float("inf")

    for employee in employees:
        task_count = len(employee.get("tasks", []))
        workload   = employee.get("workload", 0.0)

        # ML-score van het RandomForest model
        ml_score = score_employee(employee, task, role_map)

        # Exponentiële werkdruk penalty
        workload_penalty = workload * (1 + 0.5 * task_count)

        # Minimumbonus voor werknemers zonder taken
        no_task_bonus = 1.5 if task_count == 0 else 0.0

        final_score = ml_score - 0.5 * workload_penalty + no_task_bonus

        if final_score > best_score:
            best_score = final_score
            best_employee = employee

    return best_employee