import json
from allocators.task_allocator import assign_task

with open("data/employees.json") as f:
    employees = json.load(f)

with open("data/tasks.json") as f:
    tasks = json.load(f)

for task in tasks:

    employee = assign_task(task, employees)

    print("Task:", task["task"])
    print("Assigned to:", employee["name"])
    print()


#Wat we nu deden is rule-based matching. Later:
# scikit-learn
# of
# TensorFlow
# of
# PyTorch