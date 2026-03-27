import json
from ai.predictor import score_employee

def assign_task(task, employees):
    with open("models/role_map.json") as f:
        role_map = json.load(f)

    best_employee = None
    best_score = -1

    for employee in employees:
        score = score_employee(employee, task, role_map)
        if score > best_score:
            best_score = score
            best_employee = employee

    return best_employee