import json
import os
from allocators.task_allocator import assign_task

with open("data/employees.json") as f:
    employees = json.load(f)

with open("data/tasks.json") as f:
    all_tasks = json.load(f)

test_task_ids: set[str] = set()
if os.path.exists("data/test_tasks.json"):
    with open("data/test_tasks.json") as f:
        test_tasks = json.load(f)
    test_task_ids = {t["id"] for t in test_tasks}

inference_tasks = [t for t in all_tasks if t.get("id") not in test_task_ids]

print(f"Running ML allocation on {len(inference_tasks)} task(s) "
      f"({len(test_task_ids)} held out for evaluation)\n")

# init state
for emp in employees:
    emp["workload"] = emp.get("workload", 0.0)
    emp["tasks"] = []

for task in inference_tasks:
    best_employee = assign_task(task=task, employees=employees)

    best_employee["tasks"].append(task)
    best_employee["workload"] = min(1.0, best_employee["workload"] + 0.1)

    print(f"Task:        {task['task']}")
    print(f"Required:    {task['required_role']}")
    print(f"Assigned to: {best_employee['name']} ({best_employee['role']})")
    print(f"Workload:    {best_employee['workload']:.2f}")
    print(f"Tasks so far: {len(best_employee['tasks'])}")
    print()

print("=" * 40)
print("Taken per werknemer:")
print("=" * 40)
for emp in employees:
    aantal = len(emp["tasks"])
    print(f"\n{emp['name']} ({emp['role']}): {aantal} taak/taken")
    for taak in emp["tasks"]:
        print(f"  - {taak['task']}")