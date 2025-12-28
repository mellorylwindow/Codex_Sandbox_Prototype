from __future__ import annotations
from pathlib import Path
import re, csv
from datetime import date, datetime

period_re = re.compile(r"For the period\s+(?P<s>\d{2}/\d{2}/\d{4})\s+to\s+(?P<e>\d{2}/\d{2}/\d{4})", re.I)

section_markers = [
    ("deposits", re.compile(r"^Deposits and Other Additions", re.I)),
    ("withdrawals", re.compile(r"^Banking/Debit Card Withdrawals and Purchases", re.I)),
    ("checks", re.compile(r"^Checks Paid", re.I)),
    ("fees", re.compile(r"^Service Charges and Fees", re.I)),
    ("other", re.compile(r"^Other Deductions", re.I)),
]

tx_line_re = re.compile(r"^(?P<mmdd>\d{2}/\d{2})\s+(?P<amt>-?\d[\d,]*\.\d{2})\s+(?P<desc>.+)$")

def parse_mmdd(mmdd: str):
    m, d = mmdd.split("/")
    return int(m), int(d)

def pick_full_date(mmdd: str, start: date, end: date) -> date:
    m, d = parse_mmdd(mmdd)
    candidates = []
    for y in (end.year, start.year, end.year-1, end.year+1, start.year-1, start.year+1):
        try:
            candidates.append(date(y, m, d))
        except ValueError:
            pass
    inside = [c for c in candidates if start <= c <= end]
    if inside:
        return min(inside, key=lambda c: abs((end - c).days))
    return min(candidates, key=lambda c: abs((end - c).days)) if candidates else None

def infer_tx_type(section: str, amount: float) -> str:
    if section == "deposits":
        return "credit"
    if section in ("withdrawals", "fees", "checks", "other"):
        return "debit"
    return "credit" if amount < 0 else "debit"

def clean_merchant(desc: str) -> str:
    d = " ".join(desc.split())
    # strip common PNC prefixes
    for prefix in (
        "6157 Debit Card Purchase ",
        "6157 Recurring Debit Card ",
        "6157 Debit Card/Bankcard ",
        "Zel To ",
        "Zel From ",
        "Corporate ACH Payroll ",
        "Web Pmt- Deposit ",
        "Direct Payment - ",
    ):
        if d.startswith(prefix):
            d = d[len(prefix):].strip()
            break
    # drop trailing location fragments when obvious
    return d[:80].strip()

def parse_one(txt_path: Path):
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # statement period
    start = end = None
    for line in lines[:160]:
        m = period_re.search(line)
        if m:
            start = datetime.strptime(m.group("s"), "%m/%d/%Y").date()
            end   = datetime.strptime(m.group("e"), "%m/%d/%Y").date()
            break
    if not (start and end):
        start = date(2025,1,1); end = date(2025,12,31)

    # Activity Detail section scanning
    in_activity = False
    section = None
    out = []

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()

        if "Activity Detail" in line:
            in_activity = True
            i += 1
            continue
        if not in_activity:
            i += 1
            continue

        # section switching
        for name, rx in section_markers:
            if rx.match(line):
                section = name
                break

        m = tx_line_re.match(line)
        if m and section:
            mmdd = m.group("mmdd")
            amt  = float(m.group("amt").replace(",", ""))
            desc = m.group("desc").strip()

            # merge wrapped description lines:
            # keep consuming following lines that are non-empty AND do NOT start a new tx line AND do NOT look like a section header
            j = i + 1
            while j < n:
                nxt = lines[j].rstrip()
                nxt_strip = nxt.strip()
                if not nxt_strip:
                    break
                if tx_line_re.match(nxt_strip):
                    break
                if any(rx.match(nxt_strip) for _, rx in section_markers):
                    break
                # typical wrapped lines are indented or look like city/state; we just append safely
                desc += " " + nxt_strip
                j += 1

            full_dt = pick_full_date(mmdd, start, end)
            tx_type = infer_tx_type(section, amt)

            signed_amt = amt
            if tx_type == "debit" and signed_amt > 0:
                signed_amt = -signed_amt
            if tx_type == "credit" and signed_amt < 0:
                signed_amt = -signed_amt

            out.append({
                "sha": txt_path.stem,
                "statement_start": start.isoformat(),
                "statement_end": end.isoformat(),
                "date": full_dt.isoformat() if full_dt else "",
                "mmdd": mmdd,
                "section": section,
                "tx_type": tx_type,
                "amount": f"{signed_amt:.2f}",
                "description": " ".join(desc.split()),
                "merchant": clean_merchant(desc),
            })
            i = j
            continue

        i += 1

    return out

def main():
    src_dir = Path("notes/tax/work/extracted_text/2025")
    out_csv = Path("notes/tax/work/parsed/2025/pnc_spend_transactions.enriched2.csv")
    all_rows = []
    for txt in sorted(src_dir.glob("*.txt")):
        all_rows.extend(parse_one(txt))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sha","statement_start","statement_end","date","mmdd","section","tx_type","amount","merchant","description"]
        )
        w.writeheader()
        w.writerows(all_rows)

    print("wrote:", out_csv)
    print("rows:", len(all_rows))

if __name__ == "__main__":
    main()
