from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path

def main():
    matched = Path("notes/tax/work/parsed/2025/rocket_x_pnc.matched.csv")
    map_path = Path("notes/tax/work/parsed/2025/rocket_category_map.json")
    out = Path("notes/tax/work/parsed/2025/rocket_x_pnc.category_confusion.mapped.csv")

    mp = json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}

    rows = list(csv.DictReader(matched.open(newline="", encoding="utf-8")))
    c = Counter()

    for r in rows:
        rcat = (r.get("Category") or "").strip() or "<?>"
        mapped = mp.get(rcat, rcat)
        pcat = (r.get("_pnc_category") or "").strip() or "<?>"
        c[(mapped, pcat)] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rocket_category_mapped","pnc_category","match_count"])
        for (a,b),n in c.most_common():
            w.writerow([a,b,n])

    print("wrote:", out)

if __name__ == "__main__":
    main()
