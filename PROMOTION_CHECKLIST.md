cat > PROMOTION_CHECKLIST.md <<'EOF'
# Promotion Checklist (Sandbox → SwainLabs)

A sandbox artifact can be promoted only if:

- [ ] Purpose is documented (README exists)
- [ ] Runs from a fresh clone (no hidden local state)
- [ ] No hardcoded machine paths (C:\Users\..., /c/Users/..., etc.)
- [ ] Inputs/outputs are explicit
- [ ] No secrets or tokens committed
- [ ] Minimal reproduction steps included
- [ ] (Optional) Tiny test or “known-good” example included

Promotion method:
- Copy into SwainLabs_Studios as a new module
- Do NOT add SwainLabs as a dependency here
EOF
