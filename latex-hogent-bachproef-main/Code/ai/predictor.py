import joblib
from ai.dataset_builder import build_features

model = joblib.load("models/task_model.pkl")


def score_employee(employee, task, role_map):

    features = build_features(employee, task, role_map)

    score = model.predict_proba([features])[0][1] #is predict_proba een goede func?

    return score