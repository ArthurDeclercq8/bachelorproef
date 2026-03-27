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

print(f"Running allocation on {len(inference_tasks)} task(s) "
      f"({len(test_task_ids)} held out for evaluation)\n")

for task in inference_tasks:
    employee = assign_task(task, employees)
    print(f"Task:        {task['task']}")
    print(f"Required:    {task['required_role']}")
    print(f"Assigned to: {employee['name']} ({employee['role']})")
    print()