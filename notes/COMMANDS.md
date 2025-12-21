# Codex Sandbox — Commands Cheat Sheet

This repo is “offline-first”: we transcribe + generate artifacts locally, then optionally do AI-assisted rewriting later.

---

## Where files go (conventions)

### Inputs
- Media goes in: `notes/media_in/`
- Hotwords (bias the model toward correct spellings): `notes/hotwords.txt` (one phrase per line)
- Deterministic fixes: `notes/corrections.json` (string replacements)
- Redaction terms + patterns: `notes/redaction_terms.json`

### Outputs
- Transcripts: `out/transcripts/`
- Post-processed textops outputs: `out/textops/`
- Textlab runs (chunked / TOC / manifest): `out/textlab/`

---

## transcribe (main workhorse)

### Most common
transcribe demo_journal
transcribe demo_journal --srt
transcribe notes/media_in/demo_journal.mp4 --srt

### Output selection (when supported by wrapper)
- --txt (default)
- --json (segments JSON)
- --srt (subtitle file)

### Helpful flags
transcribe demo_journal --mode timestamps
transcribe demo_journal --print-latest
transcribe demo_journal --open-latest

### Quality knobs
transcribe demo_journal --language en
transcribe demo_journal --model small
transcribe demo_journal --beam-size 5
transcribe demo_journal --no-vad-filter

### Hotwords + prompt (helps proper nouns / tricky audio)
transcribe demo_journal --hotwords-file notes/hotwords.txt
transcribe demo_journal --prompt "Preserve proper nouns exactly: Duran Duran, Gizmo."
transcribe demo_journal --prompt-file notes/transcribe_prompt_journal.txt

### Post-processing (clean / redact / custom terms / report)
transcribe demo_journal --post-clean
transcribe demo_journal --post-redact standard
transcribe demo_journal --post-terms notes/redaction_terms.json
transcribe demo_journal --post-report

### Corrections pass (deterministic “post-fix”)
transcribe demo_journal --post-fix notes/corrections.json

### Raw runner (no wrapper)
python transcribe_run.py "notes/media_in/demo_journal.mp4" --mode timestamps --srt

---

## teach (build your personal fixes)

Teach updates your “memory files” so the next run improves.

### Replace a wrong phrase with the right one
teach "Duran Iran" "Duran Duran"
teach "Gismo" "Gizmo"
teach "too much behind" "two months behind"

Typical behavior:
- Adds the correct phrase to `notes/hotwords.txt` (bias)
- Adds a deterministic replacement to `notes/corrections.json` (post-fix repair)

### Verify what teach changed
cat notes/hotwords.txt
cat notes/corrections.json

---

## textops (clean + redact a transcript file)

### Basic
python textops_run.py out/transcripts/demo_journal__20251220_101645.txt --clean
python textops_run.py out/transcripts/demo_journal__*.txt --clean --redact standard --report

### Common combo
python textops_run.py out/transcripts/demo_journal__*.txt \
  --clean --clean-mode standard \
  --redact standard \
  --report

---

## textlab (chunking + TOC + manifest)

### Run it
textlab recording_10
textlab out/transcripts/recording_10__20251220_155127.txt

### Chunk sizing
textlab recording_10 --chunk-minutes 5
textlab recording_10 --chunk-minutes 2

### Print / open latest
textlab recording_10 --print-latest
textlab recording_10 --open-latest
textlab recording_10 --open-latest --open-target dir

---

## scribe (screen-recording → guide/PDF flow)

### Run scribe capture / pipeline
python scribe_run.py

### Export PDF (example)
python scribe_export_pdf.py out/scribe/<run_id>/

---

## Reading files in bash (quick recipes)

### View whole file
cat FILE

### View first / last lines
head -n 40 FILE
tail -n 40 FILE

### View a slice
sed -n '1,120p' FILE
sed -n '120,240p' FILE

### Scroll nicely
less FILE
# q to quit

### Search inside a file
grep -n "Gizmo" FILE
grep -ni "duran" FILE

### Find the newest output
ls -t out/transcripts/demo_journal__*.txt | head -n 1
ls -t out/textops/demo_journal__*.txt | head -n 1

---

## Quick “today” workflows

### 1) Transcribe + subtitles + open latest
transcribe demo_journal --srt --open-latest

### 2) Teach a fix, then rerun
teach "Duran Iran" "Duran Duran"
transcribe demo_journal --print-latest

### 3) Make it chunkable for AI later
textlab demo_journal --chunk-minutes 5 --open-latest
