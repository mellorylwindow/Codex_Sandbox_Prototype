#!/usr/bin/env bash
set -euo pipefail

./.venv_tax/Scripts/python tools/tax_match_rocket_money.py \
  --rocket notes/tax/inbox/rocket_money_export.csv \
  --pnc    notes/tax/work/parsed/2025/pnc_spend_transactions.categorized.csv \
  --out-dir notes/tax/work/parsed/2025 \
  --scope pnc_spend \
  --date-window 3 \
  --amount-tol-cents 0
