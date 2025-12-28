#!/usr/bin/env bash
set -euo pipefail

# NFCU pipeline expects these directories to exist (out/ gets git-cleaned often)
mkdir -p "out/tax_text/nfcu/2025/_meta"
mkdir -p "notes/tax/work/parsed/2025"

python tools/nfcu_extract_transactions_raw.py ${1:-}
python tools/nfcu_normalize_transactions.py
python tools/nfcu_clean_and_summarize.py
python tools/nfcu_dedupe_and_enrich.py
python tools/nfcu_step12_finalize.py
python tools/nfcu_step13_categorize.py

echo ""
echo "✅ DONE. Outputs:"
echo "  notes/tax/work/parsed/2025/nfcu_transactions.categorized.csv"
echo "  notes/tax/work/parsed/2025/nfcu_summary_by_month_category.csv"
