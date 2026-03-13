import json
import random
from sklearn.ensemble import RandomForestClassifier
from ai.dataset_builder import build_features
import joblib

with open("data/employees.json") as f:
    employees = json.load(f)

with open("data/tasks.json") as f:
    tasks = json.load(f)

X = []
y = []

for task in tasks:

    for employee in employees:

        features = build_features(employee, task)
        label = 1 if employee["role"] == task["required_role"] else 0

        X.append(features)
        y.append(label)

model = RandomForestClassifier()

model.fit(X, y)

joblib.dump(model, "models/task_model.pkl")

print("Model voltooid")