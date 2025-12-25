# Codex Sandbox — Commands Cheat Sheet

This repo is OFFLINE-FIRST: transcribe + generate artifacts locally, then optionally do AI-assisted rewriting later.

================================================================================
FOLDER MAP
================================================================================

Inputs (notes/)
  - Media:                     notes/media_in/
  - Hotwords (one per line):   notes/hotwords.txt
  - Corrections (replacements):notes/corrections.json
  - Redaction terms/patterns:  notes/redaction_terms.json
  - Optional prompts:          notes/transcribe_prompt_*.txt

Outputs (out/)
  - Transcripts:               out/transcripts/
  - Post-processed (textops):  out/textops/
  - TextLab runs (TOC/chunks): out/textlab/
  - Packs (paste-ready):       out/packs/

================================================================================
TRANSCRIBE (main workhorse)
================================================================================

Most common
  transcribe demo_journal
  transcribe demo_journal --srt
  transcribe notes/media_in/demo_journal.mp4 --srt

Output selection (when supported by wrapper)
  transcribe demo_journal --txt
  transcribe demo_journal --json
  transcribe demo_journal --srt

Useful flags
  transcribe demo_journal --mode timestamps
  transcribe demo_journal --print-latest
  transcribe demo_journal --open-latest

Quality knobs
  transcribe demo_journal --language en
  transcribe demo_journal --model small
  transcribe demo_journal --beam-size 5
  transcribe demo_journal --no-vad-filter

Hotwords + prompt (proper nouns / tricky audio)
  transcribe demo_journal --hotwords-file notes/hotwords.txt
  transcribe demo_journal --prompt "Preserve proper nouns exactly: Duran Duran, Gizmo."
  transcribe demo_journal --prompt-file notes/transcribe_prompt_journal.txt

Post-processing
  transcribe demo_journal --post-clean
  transcribe demo_journal --post-redact standard
  transcribe demo_journal --post-terms notes/redaction_terms.json
  transcribe demo_journal --post-report
  transcribe demo_journal --post-fix notes/corrections.json

Raw runner (bypass wrapper)
  python transcribe_run.py "notes/media_in/demo_journal.mp4" --mode timestamps --srt

================================================================================
TEACH (build your personal fixes)
================================================================================

Add a fix (wrong -> right)
  teach "Duran Iran" "Duran Duran"
  teach "Gismo" "Gizmo"
  teach "too much behind" "two months behind"

What teach updates
  - notes/hotwords.txt       (bias toward correct spellings)
  - notes/corrections.json   (deterministic replacement pass)

Verify
  cat notes/hotwords.txt
  cat notes/corrections.json

================================================================================
TEXTOPS (clean + redact transcript files)
================================================================================

Basic
  python textops_run.py out/transcripts/demo_journal__20251220_101645.txt --clean
  python textops_run.py out/transcripts/demo_journal__*.txt --clean --redact standard --report

Common combo
  python textops_run.py out/transcripts/demo_journal__*.txt --clean --clean-mode standard --redact standard --report

================================================================================
TEXTLAB (chunking + TOC + manifest)
================================================================================

Run it
  textlab recording_10
  textlab out/transcripts/recording_10__20251220_155127.txt

Chunk sizing
  textlab recording_10 --chunk-minutes 5
  textlab recording_10 --chunk-minutes 2

Print / open latest
  textlab recording_10 --print-latest
  textlab recording_10 --open-latest
  textlab recording_10 --open-latest --open-target dir

================================================================================
SCRIBE (screen recording -> guide/PDF flow)
================================================================================

Run capture/pipeline
  python scribe_run.py

Export PDF
  python scribe_export_pdf.py out/scribe/<run_id>/

================================================================================
CORPUSSEARCH (offline search over your exported writing)
================================================================================

Searches out/my_corpus/my_messages.jsonl

  corpussearch "velvet os"
  corpussearch "border states" --k 25
  corpussearch "tax engine" --chunk-chars 1200 --overlap 200
  corpussearch "transformus" --source "C:\Users\naked\Downloads\chatgpt-export\my_messages.jsonl"

================================================================================
CORPUSPACK (make a paste-ready Markdown pack from your writing)
================================================================================

Writes a single Markdown pack to out/packs/

  corpuspack "velvet os"
  corpuspack "border states" --k 25
  corpuspack "mallow" --include-meta
  corpuspack "tax engine" --max-chars 900
  corpuspack "transformus" --out out/packs/transformus_seed.md

================================================================================
BASH FILE READING (quick recipes)
================================================================================

View
  cat FILE
  head -n 40 FILE
  tail -n 40 FILE
  sed -n '1,120p' FILE
  sed -n '120,240p' FILE
  less FILE        (press q to quit)

Search
  grep -n  "Gizmo" FILE
  grep -ni "duran" FILE

Newest output
  ls -t out/transcripts/demo_journal__*.txt | head -n 1
  ls -t out/textops/demo_journal__*.txt     | head -n 1

================================================================================
QUICK “TODAY” WORKFLOWS
================================================================================

1) Transcribe + subtitles + open latest
  transcribe demo_journal --srt --open-latest

2) Teach a fix, then rerun
  teach "Duran Iran" "Duran Duran"
  transcribe demo_journal --print-latest

3) Make it chunkable for AI later
  textlab demo_journal --chunk-minutes 5 --open-latest
