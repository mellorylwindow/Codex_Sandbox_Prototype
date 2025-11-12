# 🧾 CHANGELOG_V9.md  
**Codex Sandbox Prototype — TikTok Emotional Function Pipeline (V9)**  
_“Every iteration learns a little more about what the prompt is really trying to say.”_

---

## 🗓 Release  
**Version:** V9  
**Date:** November 2025  
**Branch:** `feature/tiktok-extract`  
**Author:** Velvet Console / Codex Lab  

---

## 🔍 Overview  
V9 represents the first **fully integrated emotional function pipeline** for TikTok prompt extraction.  
It unifies category parsing, emotional tone detection, deduplication, and content preparation into one seamless flow — turning raw text into production-ready material for AutoCast, QuickCast, and Whisper outputs.

---

## 🧩 Major Changes  

### ✳️ Core Script Rewrite  
- Rebuilt `extract_tiktok_prompts.py` from the ground up with modular design.  
- Added **deduplication memory** (`seen_prompts.json`) to prevent repeat imports across runs.  
- Introduced **emotional-function detection** logic with new behavioral mapping:  
  - `directive` → Calls to action or control statements.  
  - `clarity` → Educational or “explain” tone.  
  - `grounding` → Breath, calm, sensory cues.  
  - `hype` → High-energy hooks.  
  - `affirmation` → Supportive or motivational phrasing.  
  - `storytelling` → Visualization or hypothetical prompts.  
  - `humor` → Self-aware, chaotic, or comic moments.  
- Refactored file scanning to include `.txt`, `.md`, `.json`, `.jsonl` input types.

---

## 🗂 Data Output Changes  

| File | Description |
|------|--------------|
| **`tiktok_prompts.csv`** | All parsed prompts with detected categories and base emotional tags. |
| **`tiktok_prompts_categorized.csv`** | Cleaned, sorted dataset used for training or AutoCast generation. |
| **`seen_prompts.json`** | Deduplication memory tracking all processed prompt hashes. |
| **`quickcast.md`** | 90-second cut scripts for social pacing. |
| **`autocast_scripts.md`** | Long-form grouped prompts formatted for AI voiceover. |
| **`autocast_whisper.json`** | Whisper JSON export for speech synthesis / caption sync. |
| **`teleprompter.html`** | On-screen scroll-ready render for creators. |

---

## ⚙️ Functional Upgrades  

- **Pattern matching overhaul** using compiled regex sets for faster parse cycles.  
- Added **group clustering** logic for natural sequence ordering.  
- Streamlined **error handling** for malformed lines and empty inputs.  
- **JSON compatibility layer** added to prepare for future GPT-based vector analysis.  
- Introduced human-legible console summaries after each run.  

Example:

---

## 🧠 Internal Refactors  
- Reorganized constants and paths (`RAW_DIR`, `OUT_DIR`, `CAT_CSV`) for better portability.  
- Separated extraction logic from CSV writing for easier testing.  
- Replaced old hard-coded categories with extensible `CATEGORY_PATTERNS` dictionary.  

---

## 🧩 Stability Notes  
- Verified compatibility with Python **3.12.0 (64-bit)**.  
- Tested on Git Bash / VS Code environment.  
- All relative paths validated inside `Codex_Sandbox_Prototype/`.  
- Safe to run multiple times — dedup logic prevents overwriting or duplicate entries.  

---

## 🧬 Known Quirks  
- Some emotional tone overlap still occurs between `clarity` and `directive`.  
- Whisper JSON output lacks duration data — reserved for V10.  
- Occasional blank rows may appear if non-UTF-8 characters exist in input text.  

---

## 🧭 Next Planned Iteration – V10  
**Focus:** Voice and render layer.  
- Integrate `gTTS` / Whisper APIs for synthetic narration.  
- Generate `.srt` captions automatically.  
- Inline waveform generation via ffmpeg.  
- Establish experimental `TikTok_AutoPublisher.py` module for analytics tracking.  

---

## 💠 Philosophy  
> *Each script is a mirror.*  
> *Each prompt a possibility.*  
> *The Codex learns not by command — but by pattern.*

---

**Commit Reference:**  
`git add .`  
`git commit -m "Finalize TikTok Emotional Function Pipeline V9 – full rewrite + emotional mapping"`  
`git push origin feature/tiktok-extract`
