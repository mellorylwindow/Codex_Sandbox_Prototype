#!/usr/bin/env bash
set -euo pipefail

YEAR="${1:-2025}"
IN_DIR="notes/tax/inbox/nfcu/${YEAR}"
OUT_DIR="notes/tax/work/parsed/${YEAR}"

echo "🧾 NFCU RUN (${YEAR})"
echo "- in_dir:  ${IN_DIR}"
echo "- out:     ${OUT_DIR}"
echo ""

mkdir -p "${OUT_DIR}"

# 1) Import / parse PDFs -> normalized + summary_by_month (raw)
./.venv_tax/Scripts/python tools/tax_import_nfcu_statements.py --in "${IN_DIR}" --year "${YEAR}"

# 2) Step 10: clean + regenerate summary_by_month from cleaned logic
./.venv_tax/Scripts/python tools/nfcu_clean_and_summarize.py --year "${YEAR}"

echo ""
echo "✅ DONE. Outputs:"
echo "  ${OUT_DIR}/nfcu_transactions.normalized.csv"
echo "  ${OUT_DIR}/nfcu_transactions.cleaned.csv"
echo "  ${OUT_DIR}/nfcu_transactions.summary_by_month.csv"
echo ""

# 3) Continue pipeline
echo "▶ tools/nfcu_dedupe_and_enrich.py"
./.venv_tax/Scripts/python tools/nfcu_dedupe_and_enrich.py

echo "▶ tools/nfcu_step12_finalize.py"
./.venv_tax/Scripts/python tools/nfcu_step12_finalize.py

echo "▶ tools/nfcu_step13_categorize.py"
./.venv_tax/Scripts/python tools/nfcu_step13_categorize.py
