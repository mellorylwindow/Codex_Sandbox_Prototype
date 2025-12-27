import json, sys, csv, datetime
from pathlib import Path

def iso(ts):
    if not ts:
        return ""
    # export times are usually unix seconds (float)
    try:
        return datetime.datetime.utcfromtimestamp(float(ts)).isoformat() + "Z"
    except Exception:
        return str(ts)

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/chatgpt_index.py <export_dir> <out_csv>")
        raise SystemExit(2)

    export_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    conv_path = export_dir / "conversations.json"
    if not conv_path.exists():
        raise FileNotFoundError(f"Missing conversations.json at: {conv_path}")

    data = json.loads(conv_path.read_text(encoding="utf-8"))
    # conversations.json is typically a dict keyed by conversation_id
    rows = []
    for cid, c in (data or {}).items():
        title = (c.get("title") or "").strip()
        create_time = c.get("create_time")
        update_time = c.get("update_time")
        mapping = c.get("mapping") or {}
        msg_count = sum(1 for _k, v in mapping.items() if (v.get("message") is not None))
        rows.append({
            "conversation_id": cid,
            "title": title,
            "create_time_utc": iso(create_time),
            "update_time_utc": iso(update_time),
            "message_count": msg_count,
        })

    rows.sort(key=lambda r: r["update_time_utc"] or "", reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["conversation_id","title","create_time_utc","update_time_utc","message_count"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} conversations -> {out_csv}")

if __name__ == "__main__":
    main()
