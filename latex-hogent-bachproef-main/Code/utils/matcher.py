def score_employee_for_task(employee, task):

    score = 0

    #Role match
    if employee["role"] == task["required_role"]:
        score += 5

    #Experience
    experience_score = employee.get("experience", 0)
    score += 0.5 * experience_score

    #Workload penalty
    workload_penalty = employee.get("workload", 0)
    score -= 2 * workload_penalty

    return score