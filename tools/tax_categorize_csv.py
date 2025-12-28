from __future__ import annotations
import csv, json, re
from pathlib import Path
from collections import Counter, defaultdict

def load_rules(path: Path):
    rules = json.loads(path.read_text(encoding="utf-8"))
    compiled = []
    for cat, pats in rules.items():
        for p in pats:
            compiled.append((cat, re.compile(re.escape(p), re.I)))
    return compiled

def categorize(desc: str, merchant: str, rules):
    hay = f"{merchant} {desc}".strip()
    for cat, rx in rules:
        if rx.search(hay):
            return cat
    return "Uncategorized"

def main():
    inp = Path("notes/tax/work/parsed/2025/pnc_spend_transactions.enriched2.csv")
    rules_path = Path("notes/tax/work/parsed/2025/category_rules.json")
    outp = Path("notes/tax/work/parsed/2025/pnc_spend_transactions.categorized.csv")

    rules = load_rules(rules_path)
    rows = list(csv.DictReader(inp.open(newline="", encoding="utf-8")))

    cat_counts = Counter()
    uncats = Counter()

    for r in rows:
        cat = categorize(r.get("description",""), r.get("merchant",""), rules)
        r["category"] = cat
        cat_counts[cat] += 1
        if cat == "Uncategorized":
            uncats[r.get("merchant","").strip() or r.get("description","")[:60]] += 1

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        if "category" not in fieldnames:
            fieldnames.append("category")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("wrote:", outp)
    print("\ncategory counts:")
    for k,v in cat_counts.most_common():
        print(f"{v:4d}  {k}")

    print("\nTop uncategorized merchants:")
    for k,v in uncats.most_common(25):
        print(f"{v:4d}  {k}")

if __name__ == "__main__":
    main()
