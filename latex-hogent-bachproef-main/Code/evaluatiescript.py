# Vergelijkt drie taakallocatiemethodes op dezelfde dataset:
#   1. Round-Robin
#   2. Greedy Rol-Match
#   3. ML-model (RandomForest)

import json, copy, statistics, sys
from collections import Counter

with open("data/employees.json") as f: employees = json.load(f)
with open("data/tasks.json")    as f: tasks     = json.load(f)

def metrics(assignments):
    role_match, skill_hits = 0, 0
    workload = {e["name"]: e.get("workload", 0.0) for e in employees}
    counts   = Counter()
    for task, emp in assignments:
        counts[emp["name"]] += 1
        workload[emp["name"]] = min(1.0, workload[emp["name"]] + 0.1)
        req = task["required_role"]
        if req == emp["role"] or (req == "Senior Developer" and emp["role"] == "Developer"):
            role_match += 1
        task_text = (task["task"] + " " + task.get("type","")).lower()
        if any(s.lower() in task_text for s in emp.get("skills",[])):
            skill_hits += 1
    n = len(assignments)
    c = [counts.get(e["name"], 0) for e in employees]
    overload = sum(1 for w in workload.values() if w >= 1.0)
    return {
        "Rol-match %":            round(100 * role_match / n, 1),
        "Skill-overlap %":        round(100 * skill_hits / n, 1),
        "Werklastverdeling std":  round(statistics.stdev(c), 2),
        "Max taken (1 persoon)":  max(c),
        "Min taken (1 persoon)":  min(c),
        "Overbelasting %":        round(100 * overload / len(employees), 1),
    }

def fresh():
    emps = copy.deepcopy(employees)
    for e in emps: e["tasks"] = []
    return emps

# Round-Robin 
emps = fresh()
rr   = [(t, emps[i % len(emps)]) for i, t in enumerate(tasks)]

# Greedy Rol-Match 
emps = fresh()
gr   = []
for task in tasks:
    req  = task["required_role"]
    pool = [e for e in emps if e["role"] == req
            or (req == "Senior Developer" and e["role"] == "Developer")] or emps
    best = min(pool, key=lambda e: (e["workload"], len(e["tasks"])))
    best["tasks"].append(task); best["workload"] = min(1.0, best["workload"] + 0.1)
    gr.append((task, best))

# ML-model 
ml = None
try:
    from allocators.task_allocator import assign_task
    emps = fresh()
    ml   = []
    for task in tasks:
        best = assign_task(task=task, employees=emps)
        best["tasks"].append(task)
        best["workload"] = min(1.0, best["workload"] + 0.1)
        ml.append((task, best))
except Exception as e:
    print(f"ML-model niet beschikbaar: {e}")
    print("Voer eerst 'python ai/train_model.py' uit.\n")

# Output per methode 
methodes = [("Round-Robin", rr), ("Greedy Rol-Match", gr)]
if ml: methodes.append(("ML-model (RandomForest)", ml))

for naam, asgn in methodes:
    print(f"\n{'='*40}\n  {naam}\n{'='*40}")
    for k, v in metrics(asgn).items():
        print(f"  {k:<28} {v}")
    cnt = Counter(e["name"] for _, e in asgn)
    print(f"\n  Taakverdeling:")
    for e in employees:
        n = cnt.get(e["name"], 0)
        print(f"  {'X'*n:<12} {n:>2}  {e['name']}")

# Vergelijkingstabel 
if ml:
    higher_is_better = {
        "Rol-match %":           True,
        "Skill-overlap %":       True,
        "Werklastverdeling std": False,
        "Max taken (1 persoon)": False,
        "Min taken (1 persoon)": True,
        "Overbelasting %":       False,
    }
    all_metrics = {n: metrics(a) for n, a in methodes}
    keys = list(next(iter(all_metrics.values())).keys())

    print(f"\n{'='*70}")
    print("  VERGELIJKINGSTABEL")
    print(f"{'='*70}")
    print(f"  {'Metric':<28}", end="")
    for naam in all_metrics: print(f"  {naam[:20]:<22}", end="")
    print()
    print(f"  {'─'*28}", end="")
    for _ in all_metrics: print(f"  {'─'*22}", end="")
    print()

    for key in keys:
        vals = {n: m[key] for n, m in all_metrics.items()}
        best = max(vals.values()) if higher_is_better[key] else min(vals.values())
        print(f"  {key:<28}", end="")
        for naam, val in vals.items():
            marker = " V" if val == best else "  "
            print(f"  {str(val)+marker:<22}", end="")
        print()
    print(f"\n  V = beste score voor die metric")