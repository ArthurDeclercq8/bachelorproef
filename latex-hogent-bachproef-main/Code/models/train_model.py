import json
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.stress_model import compute_stress

def generate_training_data(employees, tasks, n_samples=200):

    X = []
    y = []

    for _ in range(n_samples):

        employee = random.choice(employees)
        task = random.choice(tasks)

        # Features
        skill_match = len(set(employee["role"]).intersection(task["required_role"])) / max(len(task["required_role"]), 1)
        workload = employee.get("workload", random.uniform(0, 1))
        stress = compute_stress(employee, task)
        difficulty = task.get("difficulty", random.uniform(0, 1))

        features = [skill_match, workload, stress, difficulty]

        # het learning
        score = (
            0.5 * skill_match +
            0.3 * (1 - workload) +
            0.2 * (1 - stress)
        )

        X.append(features)
        y.append(score)

    return np.array(X), np.array(y)


def train_model():

    with open("data/employees.json") as f:
        employees = json.load(f)

    with open("data/tasks.json") as f:
        tasks = json.load(f)

    X, y = generate_training_data(employees, tasks)

    model = LinearRegression()
    model.fit(X, y)

    return model