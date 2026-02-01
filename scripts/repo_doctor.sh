#!/usr/bin/env bash
set -euo pipefail

echo "== Repo Doctor =="

echo
echo "-- suspicious root names (contains ':') --"
ls -1 | sed -n '/:/p' || true

echo
echo "-- paths containing spaces or single quotes (harder to script) --"
# exclude .git
find . -path './.git' -prune -o -print | grep -E "[[:space:]]|'" | head -n 200 || true

echo
echo "-- ~BROMIUM dirs (usually duplicates) --"
find . -type d -name "~BROMIUM" -print | head -n 200 || true

echo
echo "-- large files (top 30) --"
# portable-ish: uses du; on mac use gdu or adjust
find . -type f -not -path "./.git/*" -print0 \
  | xargs -0 du -h 2>/dev/null \
  | sort -hr | head -n 30 || true

echo
echo "Done."
