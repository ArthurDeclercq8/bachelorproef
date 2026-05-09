import json
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset_builder import build_role_map, build_features

PERSONA_WEIGHTS = {
    "technical":  ["technical_skill", "problem_solving", "code_quality"],
    "support":    ["communication", "reliability", "speed"],
    "management": ["collaboration", "communication", "initiative"],
    "ops":        ["reliability", "stress_handling", "focus"],
}

TASK_TYPE_TO_PERSONA = {
    "development": "technical",
    "support":     "support",
    "management":  "management",
    "devops":      "ops",
}


def _persona_score(employee: dict, task: dict) -> float:
    task_type = task.get("type", "")
    persona_key = TASK_TYPE_TO_PERSONA.get(task_type, "technical")
    traits = PERSONA_WEIGHTS[persona_key]
    persona = employee["persona"]
    values = [persona.get(t, 5) for t in traits]
    return (sum(values) / len(values)) / 10.0 


def composite_score(employee: dict, task: dict) -> float:
    role_match  = 1.0 if employee["role"] == task["required_role"] else 0.0
    persona_fit = _persona_score(employee, task)
    
    workload    = employee.get("workload", 0.0)
    task_count  = len(employee.get("tasks", []))
    
    # Exponentiële penalty: hoe meer taken, hoe zwaarder de straf
    workload_penalty = workload * (1 + 0.5 * task_count)
    
    # Minimumbonus: werknemers zonder taken worden kunstmatig aantrekkelijker
    no_task_bonus = 1.5 if task_count == 0 else 0.0

    return role_match + 0.5 * persona_fit - 0.5 * workload_penalty + no_task_bonus


def build_dataset(tasks: list, employees: list, top_k: int = 1):
    X, y = [], []

    for task in tasks:
        scores = [(emp, composite_score(emp, task)) for emp in employees]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_names = {emp["name"] for emp, _ in scores[:top_k]}

        for employee in employees:
            features = build_features(employee, task, ROLE_MAP)
            label = 1 if employee["name"] in top_names else 0
            X.append(features)
            y.append(label)

    return X, y

with open("data/employees.json") as f:
    employees = json.load(f)

with open("data/tasks.json") as f:
    tasks = json.load(f)

ROLE_MAP = build_role_map(employees, tasks)

import json
with open("models/role_map.json", "w") as f:
    json.dump(ROLE_MAP, f, indent=2)

train_tasks, test_tasks = train_test_split(tasks, test_size=0.2, random_state=42)

with open("data/test_tasks.json", "w") as f:
    json.dump(test_tasks, f, indent=2)

print(f"Train tasks: {len(train_tasks)}  |  Test tasks: {len(test_tasks)}")

X_train, y_train = build_dataset(train_tasks, employees, top_k=2)
X_test,  y_test  = build_dataset(test_tasks,  employees)

print(f"Train samples: {len(X_train)}  (positives: {sum(y_train)})")
print(f"Test  samples: {len(X_test)}   (positives: {sum(y_test)})")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["not best", "best"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

feature_names = [
    "employee_role", "task_required_role", "experience", "workload",
    "technical_skill", "problem_solving", "speed", "code_quality",
    "reliability", "focus", "collaboration", "communication",
    "stress_handling", "initiative", "difficulty", "days_remaining",
]
importances = model.feature_importances_
ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("\nTop feature importances:")
for name, imp in ranked[:8]:
    print(f"  {name:<22} {imp:.3f}")

joblib.dump(model, "models/task_model.pkl")
print("\nModel saved to models/task_model.pkl")