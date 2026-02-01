#!/usr/bin/env python3
"""
med_eob_prep.py (v3 - hard filter + hard trim + plan-year date guard)

Fixes observed issues:
- patient_name pollution (always trims to FIRST LAST; rejects junk)
- filters out non-EOB-ish pages/groups (prevents UONELNOSSY/JUNOWY/etc)
- prevents plan-year ranges (01/01–12/31) from overwriting true DOS dates
- better provider extraction (avoids address/summary/responsibility junk)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None


# ---------------------------
# Helpers
# ---------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(s: str) -> str:
    s = (s or "").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_capture_dt_from_filename(stem: str) -> str:
    m = re.match(r"^(\d{8})_(\d{6})$", stem)
    if not m:
        return ""
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.isoformat(timespec="seconds")
    except ValueError:
        return ""


def parse_date_any(s: str) -> Optional[date]:
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def money_to_float(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", "").replace("$", "")
    if s.startswith("."):
        s = "0" + s
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def score_text_quality(txt: str) -> int:
    t = txt or ""
    letters = len(re.findall(r"[A-Za-z]", t))
    digits = len(re.findall(r"\d", t))
    words = len(re.findall(r"[A-Za-z0-9]{2,}", t))
    return letters + digits + words


def looks_like_address(up: str) -> bool:
    if re.search(r"\b\d{5}(?:-\d{4})?\b", up):
        return True
    if re.search(r"\b(AVENUE|AVE|STREET|ST|ROAD|RD|DRIVE|DR|LANE|LN|APT|SUITE|STE|PO BOX)\b", up):
        return True
    if re.search(r"\b(MD|VA|DC|ND)\b", up) and re.search(r"\b\d{5}\b", up):
        return True
    return False


# ---------------------------
# OCR
# ---------------------------

def ocr_best_rotation(img_path: Path, min_len: int = 80) -> Tuple[bool, str, int, str]:
    if Image is None or pytesseract is None:
        return False, "", 0, "ocr_unavailable"

    try:
        img0 = Image.open(img_path)
    except Exception as e:
        return False, "", 0, f"ocr_open_failed:{type(e).__name__}"

    best_txt = ""
    best_rot = 0
    best_score = -1
    tried: List[str] = []

    for rot in (0, 90, 180, 270):
        try:
            img = img0.rotate(rot, expand=True) if rot else img0
            txt = norm_ws(pytesseract.image_to_string(img) or "")
            sc = score_text_quality(txt)
            tried.append(f"rotation={rot} score={sc}")
            if sc > best_score:
                best_score = sc
                best_txt = txt
                best_rot = rot
        except Exception as e:
            tried.append(f"rotation={rot} error={type(e).__name__}")

    clean = (best_txt or "").strip()
    if len(clean) < min_len:
        return False, best_txt, best_rot, f"ocr_no_text best_rotation={best_rot} best_score={best_score} len={len(clean)}; " + "; ".join(tried)[:250]

    return True, best_txt, best_rot, "; ".join(tried)[:250]


# ---------------------------
# Extraction
# ---------------------------

BCBS_JUNK_SUBSTR = (
    "BLUE CROSS", "BLUE SHIELD", "INDEPENDENT LICENSEE", "EXPLANATION OF BENEFITS",
    "HELP STOP FRAUD", "MEMBER SERVICES", "COMPLIANCE", "CONTACT:", "TTY",
    "SUMMARY", "TOTAL", "RESPONSIBILITY", "ALLOWED", "AMOUNT YOU OWE",
    "THIS IS NOT A BILL", "PAID BY OTHER INSURANCE",
)

PROVIDER_GOOD_TOKENS = (
    "MD", "DO", "DDS", "DMD", "CENTER", "CLINIC", "HOSPITAL",
    "LAB", "LABORATORY", "RADIOLOGY", "ASSOC", "ASSOCIATES", "CORP", "INC", "LLC",
    "HEALTH", "MEDICAL", "IMAGING", "PHYSICIAN", "PATIENT FIRST",
)

SERVICE_CONTEXT = ("date of service", "service date", "dos", "visit date", "from", "to")


@dataclass
class PageExtract:
    form_id: str = ""
    insurance_claim_number_raw: str = ""
    insurance_claim_number: str = ""
    claim_extract_reason: str = "no_match"
    patient_name: str = ""
    provider_name: str = ""
    service_date_min: str = ""
    service_date_max: str = ""
    date_extract_reason: str = "dates=0"
    total_billed: Optional[float] = None
    total_provider_responsibility: Optional[float] = None
    total_benefits_approved: Optional[float] = None
    amount_you_owe_provider: Optional[float] = None
    has_eob_keyword: bool = False


def extract_form_id(txt: str) -> str:
    m = re.search(r"\bH\d{6,}\b", txt)
    return m.group(0).lower() if m else ""


def extract_claim_number(txt: str) -> Tuple[str, str, str]:
    m = re.search(r"\bClaim\s*(?:Number|No\.?|ID)\s*[:#]?\s*([A-Z0-9\-]{6,})\b", txt, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        return raw, raw, "matched:ClaimNumber"
    return "", "", "no_match"


def trim_first_last(up_line: str) -> str:
    """
    Always returns exactly "FIRST LAST" if possible.
    """
    up = (up_line or "").upper().strip()
    m = re.search(r"\b([A-Z][A-Z'\-]+)\s+([A-Z][A-Z'\-]+)\b", up)
    if not m:
        return ""
    return f"{m.group(1)} {m.group(2)}"


def patient_is_junk(name2: str) -> bool:
    up = (name2 or "").upper().strip()
    if not up:
        return True
    # reject obvious non-people
    if any(k in up for k in ("TOTAL", "DATES", "SPEND", "PAGE", "INSURANCE", "BILL", "ALLOWED", "RESPONSIBILITY")):
        return True
    # reject OCR-gibberish patterns (lots of O/N/I + no vowels-ish)
    if re.fullmatch(r"[A-Z]{8,}\s+[A-Z]{4,}", up) and ("SWAIN" not in up):
        # allow normal names but this catches many garbage ones; we rely on SWAIN for your case
        return True
    return False


def extract_patient_name(txt: str) -> str:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    for ln in lines:
        up = ln.upper()
        if "SWAIN" in up and not looks_like_address(up):
            n2 = trim_first_last(up)
            if "SWAIN" in n2:
                return n2

    # fallback: first plausible non-junk 2-word name
    for ln in lines:
        up = ln.upper()
        if any(j in up for j in BCBS_JUNK_SUBSTR):
            continue
        if looks_like_address(up):
            continue
        n2 = trim_first_last(up)
        if n2 and not patient_is_junk(n2):
            return n2

    return ""


def extract_provider_name(txt: str) -> str:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    best = ""
    best_score = -10**9

    for ln in lines:
        up = ln.upper()

        if looks_like_address(up):
            continue
        if any(j in up for j in ("BLUE CROSS", "BLUE SHIELD", "INDEPENDENT LICENSEE", "HELP STOP FRAUD", "COMPLIANCE")):
            continue
        if any(j in up for j in ("SUMMARY", "TOTAL", "AMOUNT YOU OWE", "RESPONSIBILITY", "ALLOWED")):
            continue
        if re.search(r"\b(CLAIM NUMBER|PATIENT ID|GROUP NUMBER|CONTROL NUMBER)\b", up):
            continue
        if len(up) < 8:
            continue

        token_bonus = sum(1 for tok in PROVIDER_GOOD_TOKENS if tok in up) * 25
        sc = score_text_quality(up) + token_bonus

        if sc > best_score:
            best_score = sc
            best = up

    return best.strip()


def extract_service_dates(txt: str) -> Tuple[str, str, str]:
    date_pat = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"

    # 1) Prefer dates near "date of service" context
    svc: List[date] = []
    for m in re.finditer(date_pat, txt):
        d = parse_date_any(m.group(1))
        if not d:
            continue
        pos = m.start()
        window = txt[max(0, pos - 80): pos + 80].lower()
        if any(k in window for k in SERVICE_CONTEXT):
            svc.append(d)

    if svc:
        ds = sorted(set(svc))
        return ds[0].isoformat(), ds[-1].isoformat(), f"service_context_dates={len(ds)}"

    # 2) fallback: collect all dates, but do NOT allow plan-year to dominate
    all_dates: List[date] = []
    for m in re.finditer(date_pat, txt):
        d = parse_date_any(m.group(1))
        if d:
            all_dates.append(d)

    if not all_dates:
        return "", "", "dates=0"

    ds = sorted(set(all_dates))

    # detect plan-year pair
    if len(ds) >= 2:
        by_year: Dict[int, set[tuple[int, int]]] = {}
        for d in ds:
            by_year.setdefault(d.year, set()).add((d.month, d.day))
        # if we have (1,1) and (12,31) in the same year and ALSO other dates, drop the plan-year endpoints
        filtered: List[date] = []
        for d in ds:
            if (d.month, d.day) in ((1, 1), (12, 31)) and ((1, 1) in by_year.get(d.year, set()) and (12, 31) in by_year.get(d.year, set())):
                # keep only if these are the ONLY dates in this year
                pass
            else:
                filtered.append(d)

        # if filtering removed too much (meaning ds was only plan-year), revert
        if filtered:
            ds = sorted(set(filtered))

    # still too many => garbage OCR
    if len(ds) > 12:
        return "", "", f"dates_too_many={len(ds)}"

    return ds[0].isoformat(), ds[-1].isoformat(), f"dates_fallback={len(ds)}"


def extract_money_fields(txt: str) -> Dict[str, Optional[float]]:
    def grab(label: str) -> Optional[float]:
        m = re.search(rf"\b{re.escape(label)}\b\D{{0,40}}\$?\s*([0-9,]*\.?[0-9]{{0,2}})", txt, flags=re.IGNORECASE)
        if not m:
            return None
        return money_to_float(m.group(1))

    return {
        "total_billed": grab("Total Billed"),
        "total_provider_responsibility": grab("Total Provider Responsibility"),
        "total_benefits_approved": grab("Total Benefits Approved"),
        "amount_you_owe_provider": grab("Amount You Owe Provider"),
    }


def is_eobish(txt: str, ex: PageExtract) -> bool:
    up = (txt or "").upper()
    if "EXPLANATION OF BENEFITS" in up:
        return True
    if ex.insurance_claim_number:
        return True
    if ex.patient_name and "SWAIN" in ex.patient_name:
        return True
    if ex.total_billed is not None or ex.total_provider_responsibility is not None or ex.amount_you_owe_provider is not None:
        return True
    return False


def build_group_id(form_id: str, claim_num: str, sha256: str) -> str:
    if form_id:
        return form_id
    if claim_num:
        return claim_num.lower()
    return hashlib.sha1(sha256.encode("utf-8")).hexdigest()[:12]


def resolve_year_dir(root: Path, tax_year: str) -> Path:
    yd = root / str(tax_year)
    if yd.exists() and yd.is_dir():
        return yd
    return root


def find_eob_images(year_dir: Path) -> List[Path]:
    eob_dir = year_dir / "10_sources" / "insurance" / "bcbsnd" / "eob_images"
    if not eob_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted([p for p in eob_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def ensure_dirs(root: Path) -> Tuple[Path, Path]:
    idx = root / "index"
    rep = root / "40_reports"
    idx.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    return idx, rep


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tax_intake")
    ap.add_argument("--tax-year", default="2025")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--require-eob-text", action="store_true")
    ap.add_argument("--write-ocr-text", action="store_true")
    ap.add_argument("--min-ocr-len", type=int, default=80)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    year_dir = resolve_year_dir(root, args.tax_year)
    idx_dir, reports_dir = ensure_dirs(root)

    # If you request text or require text, OCR must be on
    if (args.require_eob_text or args.write_ocr_text) and not args.ocr:
        args.ocr = True

    imgs = find_eob_images(year_dir)
    if not imgs:
        raise SystemExit(f"No EOB images found under {year_dir / '10_sources/insurance/bcbsnd/eob_images'}")

    ocr_text_dir = year_dir / "10_sources" / "insurance" / "bcbsnd" / "eob_ocr_text"
    if args.write_ocr_text:
        ocr_text_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []
    page_rows: List[dict] = []

    for p in imgs:
        sha = sha256_file(p)
        capture_dt = parse_capture_dt_from_filename(p.stem)
        relpath = str(p.relative_to(year_dir)).replace("/", "\\")

        txt = ""
        rot = 0
        ocr_ok = False
        ocr_reason = "ocr_disabled"
        ocr_text_relpath = ""

        if args.ocr:
            ok, txt2, rot2, reason = ocr_best_rotation(p, min_len=args.min_ocr_len)
            ocr_ok = bool(ok)
            txt = txt2 or ""
            rot = rot2
            ocr_reason = reason

            if args.write_ocr_text:
                txt_path = ocr_text_dir / (p.stem + ".txt")
                txt_path.write_text(txt, encoding="utf-8", newline="\n")
                ocr_text_relpath = txt_path.relative_to(year_dir).as_posix().replace("/", "\\")

        if args.require_eob_text and args.ocr and not ocr_ok:
            manifest_rows.append({
                "sha256": sha,
                "relpath": relpath,
                "filename": p.name,
                "capture_dt": capture_dt,
                "ocr_ok": False,
                "ocr_reason": ocr_reason,
                "rotation": rot,
                "skipped": True,
            })
            continue

        ex = PageExtract()
        if txt.strip():
            ex.has_eob_keyword = ("EXPLANATION OF BENEFITS" in txt.upper())

            ex.form_id = extract_form_id(txt)
            raw, norm, reason = extract_claim_number(txt)
            ex.insurance_claim_number_raw = raw
            ex.insurance_claim_number = norm
            ex.claim_extract_reason = reason

            ex.patient_name = extract_patient_name(txt)
            # hard trim again just in case
            if ex.patient_name:
                ex.patient_name = trim_first_last(ex.patient_name)

            ex.provider_name = extract_provider_name(txt)

            dmin, dmax, dreason = extract_service_dates(txt)
            ex.service_date_min = dmin
            ex.service_date_max = dmax
            ex.date_extract_reason = dreason

            money = extract_money_fields(txt)
            ex.total_billed = money["total_billed"]
            ex.total_provider_responsibility = money["total_provider_responsibility"]
            ex.total_benefits_approved = money["total_benefits_approved"]
            ex.amount_you_owe_provider = money["amount_you_owe_provider"]

        # FILTER OUT GARBAGE PAGES:
        # If it doesn't look like an EOB, don't even include it in pages -> groups.
        if txt.strip() and not is_eobish(txt, ex):
            manifest_rows.append({
                "sha256": sha,
                "relpath": relpath,
                "filename": p.name,
                "capture_dt": capture_dt,
                "ocr_ok": bool(ocr_ok),
                "ocr_reason": ocr_reason,
                "rotation": rot,
                "filtered_out": True,
            })
            continue

        manifest_rows.append({
            "sha256": sha,
            "relpath": relpath,
            "filename": p.name,
            "capture_dt": capture_dt,
            "ocr_ok": bool(ocr_ok),
            "ocr_reason": ocr_reason,
            "rotation": rot,
        })

        page_rows.append({
            "sha256": sha,
            "relpath": relpath,
            "filename": p.name,
            "form_id": ex.form_id,
            "insurance_claim_number_raw": ex.insurance_claim_number_raw or "",
            "insurance_claim_number": ex.insurance_claim_number or "",
            "claim_extract_reason": ex.claim_extract_reason,
            "patient_name": ex.patient_name or "",
            "provider_name": ex.provider_name or "",
            "service_date_min": ex.service_date_min or "",
            "service_date_max": ex.service_date_max or "",
            "date_extract_reason": ex.date_extract_reason,
            "total_billed": ex.total_billed,
            "total_provider_responsibility": ex.total_provider_responsibility,
            "total_benefits_approved": ex.total_benefits_approved,
            "amount_you_owe_provider": ex.amount_you_owe_provider,
            "has_eob_keyword": bool(ex.has_eob_keyword),
            "ocr_ok": bool(ocr_ok),
            "ocr_reason": ocr_reason,
            "ocr_text_relpath": ocr_text_relpath,
        })

    manifest_path = idx_dir / f"medical_eob_docs_{args.tax_year}.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    df_pages = pd.DataFrame(page_rows)
    if df_pages.empty:
        raise SystemExit("No EOB pages survived filtering. (Everything looked non-EOB-ish.)")

    df_pages["eob_group_id"] = df_pages.apply(
        lambda r: build_group_id(str(r.get("form_id", "")).strip(),
                                 str(r.get("insurance_claim_number", "")).strip(),
                                 str(r.get("sha256", "")).strip()),
        axis=1
    )

    grouped_rows: List[dict] = []
    claim_found = 0

    for gid, g in df_pages.groupby("eob_group_id", sort=True):
        g = g.sort_values("filename", ascending=True)

        # group patient: prefer SWAIN pages; always trim to FIRST LAST
        patients = [trim_first_last(x) for x in g["patient_name"].astype(str).tolist() if x and x.lower() != "nan"]
        patients = [p for p in patients if p and not patient_is_junk(p)]
        swain = [p for p in patients if "SWAIN" in p]
        patient = (max(set(swain), key=swain.count) if swain else (max(set(patients), key=patients.count) if patients else ""))

        # group provider: choose first nonblank provider-like
        providers = [x.strip().upper() for x in g["provider_name"].astype(str).tolist() if x and x.lower() != "nan"]
        providers = [p for p in providers if p and not looks_like_address(p)]
        provider = providers[0] if providers else ""

        # group dates: prefer service_context pages, then fallback, but never plan-year dominates
        date_rows = g.copy()
        svc_pref = date_rows[date_rows["date_extract_reason"].astype(str).str.contains("service_context", case=False, na=False)]
        date_source = svc_pref if len(svc_pref) else date_rows

        cand: List[date] = []
        for col in ("service_date_min", "service_date_max"):
            for v in date_source[col].astype(str).tolist():
                d = parse_date_any(v)
                if d:
                    cand.append(d)

        # drop plan-year endpoints if other dates exist
        cand = sorted(set(cand))
        if cand:
            # if we have jan1 & dec31 same year AND any other date, drop jan1/dec31
            years = {}
            for d in cand:
                years.setdefault(d.year, set()).add((d.month, d.day))
            filtered = []
            for d in cand:
                has_plan_year = ((1, 1) in years.get(d.year, set()) and (12, 31) in years.get(d.year, set()))
                if has_plan_year and (d.month, d.day) in ((1, 1), (12, 31)) and len(cand) > 2:
                    continue
                filtered.append(d)
            if filtered:
                cand = filtered

        if cand:
            service_min = cand[0].isoformat()
            service_max = cand[-1].isoformat()
            date_reason = f"group_dates={len(cand)}"
        else:
            service_min = ""
            service_max = ""
            date_reason = "dates=0"

        # claim number
        claim_num = ""
        for v in g["insurance_claim_number"].astype(str).tolist():
            v = (v or "").strip()
            if v and v.lower() != "nan":
                claim_num = v
                break
        if claim_num:
            claim_found += 1

        rep = g.iloc[0].to_dict()

        grouped_rows.append({
            "eob_group_id": gid,
            "page_count": int(len(g)),
            "pages_relpaths": " | ".join(g["relpath"].astype(str).tolist()),
            "sha256": rep.get("sha256", ""),
            "relpath": rep.get("relpath", ""),
            "filename": rep.get("filename", ""),
            "insurance_claim_number_raw": str(rep.get("insurance_claim_number_raw", "") or ""),
            "insurance_claim_number": str(claim_num or ""),
            "claim_extract_reason": str(rep.get("claim_extract_reason", "") or ""),
            "patient_name": patient,
            "provider_name": provider,
            "service_date_min": service_min,
            "service_date_max": service_max,
            "date_extract_reason": date_reason,
            "total_billed": g["total_billed"].dropna().iloc[0] if g["total_billed"].dropna().shape[0] else None,
            "total_provider_responsibility": g["total_provider_responsibility"].dropna().iloc[0] if g["total_provider_responsibility"].dropna().shape[0] else None,
            "total_benefits_approved": g["total_benefits_approved"].dropna().iloc[0] if g["total_benefits_approved"].dropna().shape[0] else None,
            "amount_you_owe_provider": g["amount_you_owe_provider"].dropna().iloc[0] if g["amount_you_owe_provider"].dropna().shape[0] else None,
            "ocr_ok": bool(g["ocr_ok"].fillna(False).any()),
            "ocr_reason": "; ".join(sorted(set([str(x) for x in g["ocr_reason"].dropna().astype(str).tolist() if x.strip()])))[:250],
            "ocr_text_relpath": str(rep.get("ocr_text_relpath", "") or ""),
        })

    df_docs = pd.DataFrame(grouped_rows)

    # Force these as strings so Excel/pandas stop "helping"
    for c in ("insurance_claim_number", "insurance_claim_number_raw", "eob_group_id"):
        if c in df_docs.columns:
            df_docs[c] = df_docs[c].fillna("").astype(str)

    csv_path = reports_dir / f"medical_eob_docs_{args.tax_year}.csv"
    df_docs.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\r\n")

    master_xlsx = reports_dir / f"medical_master_{args.tax_year}.xlsx"
    if not master_xlsx.exists():
        raise SystemExit(f"Missing medical master workbook: {master_xlsx}")

    with pd.ExcelWriter(master_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
        df_docs.to_excel(xw, index=False, sheet_name="EOBDocs")

    print(f"OK EOB manifest: {manifest_path}")
    print(f"OK EOB docs CSV: {csv_path}")
    print(f"OK Updated workbook sheet: EOBDocs -> {master_xlsx}")
    print(f"OK rows={len(df_docs)} ocr_enabled={bool(args.ocr)} insurance_claim_numbers_found={claim_found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
