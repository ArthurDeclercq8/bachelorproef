def build_features(employee, task):

    features = []
    role_match = 1 if employee["role"] == task["required_role"] else 0
    features.append(role_match)
    features.append(employee.get("experience", 0))
    features.append(employee.get("workload", 0))
    persona = employee["persona"]

    features.append(persona["technical_skill"])
    features.append(persona["problem_solving"])
    features.append(persona["speed"])
    features.append(persona["code_quality"])
    features.append(persona["reliability"])
    features.append(persona["focus"])
    features.append(persona["collaboration"])
    features.append(persona["communication"])
    features.append(persona["stress_handling"])
    features.append(persona["initiative"])

    features.append(task["difficulty"])
    features.append(task["days_remaining"])

    return features