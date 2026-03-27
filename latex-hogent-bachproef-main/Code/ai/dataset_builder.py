def build_role_map(employees: list, tasks: list) -> dict:
    roles = set()
    for e in employees:
        roles.add(e["role"])
    for t in tasks:
        roles.add(t["required_role"])
    return {role: i for i, role in enumerate(sorted(roles))}

def encode_role(role: str, role_map: dict) -> int:
    return role_map.get(role, -1)


def build_features(employee: dict, task: dict, role_map: dict) -> list:
    features = []
    features.append(encode_role(employee["role"], role_map))
    features.append(encode_role(task["required_role"], role_map))

    features.append(employee.get("experience", 0))
    features.append(employee.get("workload", 0))

    persona = employee["persona"]
    for trait in [
        "technical_skill", "problem_solving", "speed", "code_quality",
        "reliability", "focus", "collaboration", "communication",
        "stress_handling", "initiative",
    ]:
        features.append(persona[trait])

    features.append(task["difficulty"])
    features.append(task["days_remaining"])

    return features