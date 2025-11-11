import csv, json, re
from pathlib import Path

RAW_DIR = Path("data/tiktok_archive/raw")
OUT_DIR = Path("data/tiktok_archive/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "tiktok_prompts.csv"

PROMPT_PATTERNS = [
    # Lines that begin with a label
    r'^\s*(?:Prompt|Hook|CTA|Caption)\s*[:\-–]\s*(.+)$',
    # Markdown bullets that look prompt-y
    r'^\s*[-*]\s+(?:Prompt|Hook|CTA)\s*[:\-–]\s*(.+)$',
    # Bare lines in “quote” style
    r'^\s*["“](.+?)["”]\s*$',
]

compiled = [re.compile(p, re.IGNORECASE) for p in PROMPT_PATTERNS]
rows = []

def push_row(source, text, kind):
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        rows.append({"source": source, "type": kind, "text": text})

def from_text(path: Path):
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for rx in compiled:
            m = rx.match(line)
            if m:
                push_row(str(path), m.group(1), "text")
                break

def from_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    # Accept JSON or JSONL shapes
    items = data if isinstance(data, list) else [data]
    for item in items:
        # common fields people use
        for key in ("prompt", "hook", "caption", "script", "idea", "cta"):
            val = item.get(key)
            if isinstance(val, str):
                push_row(str(path), val, key)

def from_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line=line.strip()
        if not line: 
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for key in ("prompt", "hook", "caption", "script", "idea", "cta"):
            val = obj.get(key)
            if isinstance(val, str):
                push_row(str(path), val, key)

def from_md(path: Path):
    # scrape fenced code blocks and bullets first, then fall back to generic text scan
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # bullets labelled prompt/hook/cta
    for m in re.finditer(r'^\s*[-*]\s*(?:Prompt|Hook|CTA)[:\-–]\s*(.+)$', txt, flags=re.IGNORECASE|re.MULTILINE):
        push_row(str(path), m.group(1), "md")
    # generic text catch-all
    for line in txt.splitlines():
        for rx in compiled:
            mm = rx.match(line)
            if mm:
                push_row(str(path), mm.group(1), "md")
                break

def process_file(path: Path):
    ext = path.suffix.lower()
    try:
        if ext in (".txt",):
            from_text(path)
        elif ext in (".md", ".markdown"):
            from_md(path)
        elif ext == ".json":
            from_json(path)
        elif ext in (".jsonl", ".ndjson"):
            from_jsonl(path)
        # else: ignore (images, pdfs, etc.)
    except Exception as e:
        push_row(str(path), f"[parse error: {e}]", "error")

def main():
    for p in RAW_DIR.rglob("*"):
        if p.is_file():
            process_file(p)
    # de-dupe on text
    seen = set()
    deduped = []
    for r in rows:
        key = r["text"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","type","text"])
        w.writeheader()
        w.writerows(deduped)
    print(f"Wrote {len(deduped)} prompts -> {OUT_CSV}")

if __name__ == "__main__":
    main()
