def compute_stress(employee, task):
    workload = employee.get("workload", 0.5)
    difficulty = task.get("difficulty", 0.5)
    skill_match = len(set(employee["role"]).intersection(task["required_role"])) / max(len(task["required_role"]), 1)
    mismatch = 1 - skill_match
    stress = (
        0.4 * workload +
        0.3 * difficulty +
        0.3 * mismatch
    )

    return min(stress, 1.0)  