from models.train_model import train_model
from utils.stress_model import compute_stress

model = train_model()

def extract_features(employee, task):

    skill_match = len(set(employee["role"]).intersection(task["required_role"])) / max(len(task["required_role"]), 1)
    workload = employee.get("workload", 0.5)

    # dynamisch
    stress = compute_stress(employee, task)

    difficulty = task.get("difficulty", 0.5)

    return [[skill_match, workload, stress, difficulty]]


def assign_task(task, employees, model, extract_features):

    best_employee = None
    best_score = -1

    for employee in employees:

        features = extract_features(employee, task)
        score = model.predict(features)[0]

        if score > best_score:
            best_score = score
            best_employee = employee

    return best_employee