import json
from collections import Counter
from pathlib import Path

path = Path("output/final_dataset.coco.json")
data = json.loads(path.read_text())

id_to_name = {c["id"]: c["name"] for c in data["categories"]}
counts = Counter(id_to_name[a["category_id"]] for a in data["annotations"])
total = sum(counts.values())

print(f"Images: {len(data['images'])}")
print(f"Total annotations: {total}\n")
for name in id_to_name.values():
    n = counts.get(name, 0)
    pct = (n / total * 100) if total else 0
    print(f"{name:<15} {n:>6}  ({pct:.1f}%)")