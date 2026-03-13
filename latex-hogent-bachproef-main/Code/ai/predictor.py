import joblib
from ai.dataset_builder import build_features

model = joblib.load("models/task_model.pkl")


def score_employee(employee, task):

    features = build_features(employee, task)

    score = model.predict_proba([features])[0][1]

    return score