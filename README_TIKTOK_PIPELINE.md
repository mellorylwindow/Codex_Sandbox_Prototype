# 🎬 TikTok Prompt Extraction & Emotional Function Pipeline  
**Codex Sandbox Prototype – V9 (November 2025)**  
*“Turning scattered sparks into structured scripts.”*

---

## 🧭 Overview
This pipeline transforms raw TikTok prompt ideas (from `.txt`, `.md`, `.json`, `.jsonl`) into structured CSVs, categorized emotion-function tags, and generated content bundles for AutoCast, QuickCast, and Whisper voiceover sync.  

It’s part of **The Velvet Console** under the `feature/tiktok-extract` branch — a testbed for creative-automation workflows bridging text prompts → emotional function → voiceover-ready scripts.

---

## ⚙️ Architecture

Codex_Sandbox_Prototype/
├── scripts/
│ ├── extract_tiktok_prompts.py ← main extractor / categorizer
│ └── setup_check.txt ← sanity check / environment probe
├── data/
│ └── tiktok_archive/
│ ├── raw/ ← seed / input text files
│ └── derived/ ← generated outputs
├── README_TIKTOK_PIPELINE.md ← this file
├── CHANGELOG_V9.md ← version log
└── python-3.12.0-amd64.exe ← local runtime installer

### 📂 Directory Summary

| Folder / File | Description |
|----------------|-------------|
| **scripts/** | Core logic modules for extraction, categorization, and setup verification. |
| ├── `extract_tiktok_prompts.py` | Main engine that reads raw prompts, categorizes them, detects emotional function, and outputs CSVs. |
| └── `setup_check.txt` | Sanity test file to verify paths and Python setup. |
| **data/** | Houses all raw input and generated derivative data. |
| └── `tiktok_archive/` | Dedicated workspace for TikTok prompt handling. |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `raw/` | Place `.txt`, `.md`, `.json`, or `.jsonl` source prompt files here. |
| &nbsp;&nbsp;&nbsp;&nbsp;└── `derived/` | Stores CSVs, grouped JSONs, AutoCast/QuickCast outputs, and dedup memory. |
| **README_TIKTOK_PIPELINE.md** | This documentation file — describes the full data flow and emotional function mapping system. |
| **CHANGELOG_V9.md** | Records evolution across major script versions (V1 → V9). |
| **python-3.12.0-amd64.exe** | Local Python runtime installer used for environment consistency. |

---

These folders work together as a **closed creative system** — from raw text in `raw/` through pattern detection, emotional labeling, and CSV generation in `derived/`. Each iteration preserves past work, ensuring no duplicate prompts and traceable evolution across versions.
