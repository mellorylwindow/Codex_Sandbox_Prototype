from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

# Reuse-ish patterns (keep local so this script is standalone)
MONEY_RE = re.compile(r"(?<!\w)(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})|\$?\s*\d+(?:\.\d{2}))(?!\w)")
DATE_RES = [
    re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?!\d)"),   # MM/DD/YYYY
    re.compile(r"(?<!\d)(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?!\d)"),     # YYYY-MM-DD
]
PAGE_SPLIT_RE = re.compile(r"\n\s*===== PAGE (\d+)\s*=====\s*\n", re.IGNORECASE)

KEY_DATE = ("DATE OF SERVICE", "SERVICE DATE", "DOS", "VISIT DATE", "DATE:")
KEY_AMT = ("TOTAL", "AMOUNT", "PAID", "CHARGE", "BALANCE DUE", "DUE", "PAYMENT")

def norm_money(s: str) -> Optional[float]:
    s = s.strip().replace("$", "").replace(" ", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def parse_date_any(text: str) -> Optional[str]:
    for rex in DATE_RES:
        for m in rex.finditer(text):
            g = m.groups()
            try:
                if len(g) == 3 and len(g[0]) == 4:
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                else:
                    mo, d = int(g[0]), int(g[1])
                    y = int(g[2])
                    if y < 100:
                        y += 2000 if y < 70 else 1900
                return datetime(y, mo, d).strftime("%Y-%m-%d")
            except Exception:
                continue
    return None

def guess_provider(text: str) -> Optional[str]:
    bad = ("INVOICE", "RECEIPT", "STATEMENT", "THANK YOU", "TOTAL", "AMOUNT", "DATE", "PAGE")
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        u = s.upper()
        if any(b in u for b in bad) and len(s) < 40:
            continue
        if len(s) > 2:
            return s[:120]
    return None

def split_pages(text: str) -> List[Tuple[Optional[int], str]]:
    """
    Returns list of (page_number, page_text).
    If no page markers exist, returns [(None, text)].
    """
    if "===== PAGE" not in text:
        return [(None, text)]

    parts = PAGE_SPLIT_RE.split(text)
    # split yields: [pre, page1num, page1text, page2num, page2text, ...]
    out: List[Tuple[Optional[int], str]] = []
    if parts and parts[0].strip():
        out.append((None, parts[0]))
    i = 1
    while i + 1 < len(parts):
        try:
            pno = int(parts[i])
        except Exception:
            pno = None
        ptxt = parts[i + 1]
        out.append((pno, ptxt))
        i += 2
    return [(p, t.strip()) for (p, t) in out if t.strip()]

def find_best_amount_near(lines: List[str], start_idx: int) -> Optional[float]:
    """
    Look forward a few lines for the most 'total-ish' amount, else max amount.
    """
    window = lines[start_idx:start_idx + 12]
    candidates: List[Tuple[float, str]] = []
    totalish: List[Tuple[float, str]] = []

    for ln in window:
        for m in MONEY_RE.findall(ln):
            val = norm_money(m)
            if val is None:
                continue
            candidates.append((val, ln))
            u = ln.upper()
            if any(k in u for k in KEY_AMT):
                totalish.append((val, ln))

    if totalish:
        return max(totalish, key=lambda x: x[0])[0]
    if candidates:
        return max(candidates, key=lambda x: x[0])[0]
    return None

def guess_service_desc(lines: List[str], i: int) -> str:
    """
    Very simple: use the line containing the date, or next non-empty line.
    """
    base = lines[i].strip()
    if base and len(base) <= 140:
        return base
    for j in range(i + 1, min(i + 6, len(lines))):
        s = lines[j].strip()
        if s and len(s) <= 140:
            return s
    return ""

@dataclass
class LineItem:
    line_id: str
    doc_id: str
    page: Optional[int]
    provider_guess: str
    service_date_guess: str
    amount_guess: Optional[float]
    service_desc_guess: str
    confidence: str

def main() -> int:
    ap = argparse.ArgumentParser(description="Parse extracted text into line-items per document.")
    ap.add_argument("--manifest", default="tax_intake/index/manifest.jsonl")
    ap.add_argument("--text-dir", default="tax_intake/30_extracted/text")
    ap.add_argument("--out-dir", default="tax_intake/30_extracted/lines")
    ap.add_argument("--only-canonical", action="store_true")
    ap.add_argument("--force", action="store_true", help="Rebuild lines json even if it exists.")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}")
        return 2

    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    made = 0
    for r in rows:
        if args.only_canonical and r.get("duplicate_of"):
            continue

        doc_id = (r.get("doc_id") or "").strip()
        if not doc_id:
            continue

        txt_path = text_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            continue

        out_path = out_dir / f"{doc_id}.lines.json"
        if out_path.exists() and not args.force:
            continue

        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            out_path.write_text("[]\n", encoding="utf-8")
            made += 1
            continue

        provider = guess_provider(text) or ""
        pages = split_pages(text)

        items: List[LineItem] = []
        idx = 0

        for (page_no, ptxt) in pages:
            lines = [ln.rstrip() for ln in ptxt.splitlines() if ln.strip()]
            if not lines:
                continue

            # Find candidate date lines (or keyword lines)
            date_line_idxs: List[int] = []
            for i, ln in enumerate(lines):
                u = ln.upper()
                if any(k in u for k in KEY_DATE) or parse_date_any(ln):
                    # only treat as date line if there's a parseable date nearby
                    if parse_date_any(ln) or parse_date_any(" ".join(lines[i:i+2])):
                        date_line_idxs.append(i)

            # If we don't find multiple, we still produce 1 item (doc-level)
            if not date_line_idxs:
                # doc-level amount/date
                dt = parse_date_any(ptxt) or ""
                # amount: scan entire page for total-ish, else max
                page_amount = None
                totalish = []
                allcands = []
                for ln in lines[:250]:
                    for m in MONEY_RE.findall(ln):
                        val = norm_money(m)
                        if val is None:
                            continue
                        allcands.append(val)
                        if any(k in ln.upper() for k in KEY_AMT):
                            totalish.append(val)
                if totalish:
                    page_amount = max(totalish)
                elif allcands:
                    page_amount = max(allcands)

                conf = "med" if (dt or page_amount is not None) else "low"
                items.append(LineItem(
                    line_id=f"{doc_id}:{idx:03d}",
                    doc_id=doc_id,
                    page=page_no,
                    provider_guess=provider,
                    service_date_guess=dt,
                    amount_guess=page_amount,
                    service_desc_guess="",
                    confidence=conf,
                ))
                idx += 1
                continue

            # Build one item per found date line
            for i in date_line_idxs:
                block = "\n".join(lines[i:i+12])
                dt = parse_date_any(block) or parse_date_any(lines[i]) or ""
                amt = find_best_amount_near(lines, i)
                desc = guess_service_desc(lines, i)

                score = 0
                score += 1 if provider else 0
                score += 1 if dt else 0
                score += 1 if amt is not None else 0
                conf = {3: "high", 2: "med", 1: "low", 0: "none"}[score]

                items.append(LineItem(
                    line_id=f"{doc_id}:{idx:03d}",
                    doc_id=doc_id,
                    page=page_no,
                    provider_guess=provider,
                    service_date_guess=dt,
                    amount_guess=amt,
                    service_desc_guess=desc,
                    confidence=conf,
                ))
                idx += 1

        out_path.write_text(
            json.dumps([asdict(x) for x in items], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        made += 1

    print(f"Wrote line-items for {made} documents into: {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
