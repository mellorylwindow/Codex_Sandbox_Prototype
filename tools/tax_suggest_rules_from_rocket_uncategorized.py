from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_MATCHED = Path("notes/tax/work/parsed/2025/rocket_x_pnc.matched.csv")
DEFAULT_MAP = Path("notes/tax/work/parsed/2025/rocket_category_map.json")
DEFAULT_OUTDIR = Path("notes/tax/work/parsed/2025")
DEFAULT_RULES = Path("notes/tax/work/parsed/2025/category_rules.json")

PNC_UNCAT = "Uncategorized"

# --- cleaning helpers ---------------------------------------------------------

WS_RE = re.compile(r"\s+")
MASKED_RE = re.compile(r"x{6,}\d{0,6}", re.I)   # xxxxxxxx...7055 etc
LONG_DIGITS_RE = re.compile(r"\b\d{4,}\b")      # 4+ digit chunks (often ids)
DATEY_RE = re.compile(r"\b\d{4}\b")             # 4-digit tokens (often mmdd/ids)

PREFIX_STRIP = [
    r"^tst\*\s*",
    r"^sq\s*\*\s*",
    r"^py\s*\*\s*",
    r"^pos\s+purchase\s+",
    r"^pos\s+",
    r"^ach\s+",
    r"^web\s+pmt[-\s]*",
    r"^vend\s+park\s*",
]

TRAILING_NOISE = [
    # common card/location tails
    r"\s+vis\b.*$",
    r"\s+debit\b.*$",
    r"\s+credit\b.*$",
    r"\s+recurring\b.*$",
    r"\s+pos\s+purchase\b.*$",
]

# phrases that are too generic to be useful as rules
BAD_PHRASES = {
    "payment", "online payment", "pos purchase", "debit card purchase",
    "recurring", "purchase", "deposit", "atm deposit", "domestic incoming wire",
}

def norm(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())

def strip_prefixes(s: str) -> str:
    out = s
    for pat in PREFIX_STRIP:
        out = re.sub(pat, "", out, flags=re.I)
    return out.strip()

def strip_trailing_noise(s: str) -> str:
    out = s
    for pat in TRAILING_NOISE:
        out = re.sub(pat, "", out, flags=re.I)
    return out.strip()

def clean_phrase(raw: str) -> str:
    """
    Turn a noisy bank merchant/desc into a stable substring pattern.
    We aim for something you'd feel safe pasting into category_rules.json.
    """
    s = norm(raw)
    if not s:
        return ""

    s = strip_prefixes(s)
    s = strip_trailing_noise(s)

    # remove masked numbers + long digit chunks
    s = MASKED_RE.sub("", s)
    s = LONG_DIGITS_RE.sub("", s)

    s = norm(s)

    # keep it short-ish
    if len(s) > 52:
        s = s[:52].rstrip()

    # de-generic
    low = s.lower()
    if low in BAD_PHRASES:
        return ""

    # avoid suggestions that are basically empty
    if len(s) < 4:
        return ""

    return s

def pick_pnc_text(r: dict) -> str:
    # prefer explicit PNC merchant column; fall back to description
    for k in ("_pnc_merchant", "merchant", "pnc_merchant"):
        if r.get(k):
            return str(r.get(k))
    for k in ("_pnc_description", "description", "pnc_description"):
        if r.get(k):
            return str(r.get(k))
    return ""

def load_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    ap.add_argument("--map", dest="map_path", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--top_per_category", type=int, default=30)
    ap.add_argument("--write_patch_json", action="store_true", help="Write category->patterns json for easy merge")
    ap.add_argument("--autopatch_rules", action="store_true", help="Actually merge into category_rules.json (creates .bak)")
    ap.add_argument("--rules_path", type=Path, default=DEFAULT_RULES)
    args = ap.parse_args()

    rows = load_csv(args.matched)
    cat_map = load_json(args.map_path)

    # find the column names we need (tolerate variations)
    def get_pnc_cat(r: dict) -> str:
        for k in ("_pnc_category", "pnc_category", "category_pnc", "category"):
            v = (r.get(k) or "").strip()
            # in matched.csv, "category" might be Rocket's category — so only trust *_pnc_* first
            if k.startswith("_pnc_") and v:
                return v
        # fallback for older exports
        return (r.get("pnc_category") or "").strip()

    def get_rocket_cat(r: dict) -> str:
        for k in ("Category", "rocket_category"):
            if r.get(k):
                return str(r.get(k)).strip()
        return ""

    uncat = [r for r in rows if get_pnc_cat(r) == PNC_UNCAT]

    # build (target_category -> phrase counts)
    by_target: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[Tuple[str, str], str] = {}  # (target, phrase) -> example text

    skipped_no_map = 0
    skipped_empty_phrase = 0

    for r in uncat:
        rcat = get_rocket_cat(r)
        target = (cat_map.get(rcat) or "").strip()
        if not target:
            skipped_no_map += 1
            continue

        pnc_text = pick_pnc_text(r)
        phrase = clean_phrase(pnc_text)
        if not phrase:
            skipped_empty_phrase += 1
            continue

        by_target[target][phrase] += 1
        examples.setdefault((target, phrase), norm(pnc_text)[:120])

    args.outdir.mkdir(parents=True, exist_ok=True)

    # write CSV
    out_csv = args.outdir / "rocket_uncat_rule_suggestions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["suggested_target_category", "pattern", "supporting_tx_count", "example_pnc_text"])
        for target, ctr in sorted(by_target.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
            for phrase, cnt in ctr.most_common(args.top_per_category):
                if cnt < args.min_count:
                    continue
                w.writerow([target, phrase, cnt, examples.get((target, phrase), "")])

    # write JSON suggestions (category -> list of patterns)
    out_json = args.outdir / "rocket_uncat_rule_suggestions.json"
    if args.write_patch_json:
        payload = {}
        for target, ctr in by_target.items():
            pats = [p for p, c in ctr.most_common(args.top_per_category) if c >= args.min_count]
            payload[target] = pats
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # optional: merge straight into category_rules.json
    if args.autopatch_rules:
        rules = load_json(args.rules_path)
        bak = args.rules_path.with_suffix(args.rules_path.suffix + ".bak")
        bak.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        for target, ctr in by_target.items():
            rules.setdefault(target, [])
            for phrase, cnt in ctr.most_common(args.top_per_category):
                if cnt < args.min_count:
                    continue
                if phrase not in rules[target]:
                    rules[target].append(phrase)

        args.rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("PNC Uncategorized rows (matched):", len(uncat))
    print("targets found:", len(by_target))
    print("skipped (no map):", skipped_no_map)
    print("skipped (empty phrase):", skipped_empty_phrase)
    print("wrote:", out_csv)
    if args.write_patch_json:
        print("wrote:", out_json)
    if args.autopatch_rules:
        print("patched rules:", args.rules_path, "(backup:", bak, ")")

if __name__ == "__main__":
    main()
