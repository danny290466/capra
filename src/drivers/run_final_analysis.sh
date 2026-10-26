#!/usr/bin/env bash
# End-to-end analysis orchestrator: waits for every artefact, enforces the consistency gate, then
# regenerates ALL tables/numbers from the artefacts (never hand-typed).
#
# Chain: GPU pool (3 seeds) -> extra seeds 45/46 -> textdetox pair (5 seeds each)
#        + 3 LLM moderator dirs synced from the Mac -> gate -> every table.
set -u
cd .
PY=$HOME/miniconda3/envs/toxkgvi-gpu-ft/bin/python
DETS=(outputs/killtest outputs/pool/xlmr_large outputs/pool/specialist_pair
      outputs/pool/llm_deepseek_flash outputs/pool/llm_qwen38_flash outputs/pool/llm_gemini31_flashlite)

echo "=== orchestrator start $(date -Is) ==="
echo $$ > /tmp/selective_parity_orchestrator.pid
trap 'rm -f /tmp/selective_parity_orchestrator.pid' EXIT

# ---- STAGE A: everything that only needs the EN/VI pool (5 seeds) + the LLM moderators
while ! grep -q "POOL_EXTRA_DONE" logs/pool_extra.log 2>/dev/null; do
  echo "[$(date -Is)] stage A: waiting for the EN/VI pool + extra seeds..."; sleep 60
done
echo "[$(date -Is)] stage A: EN/VI pool complete (5 seeds)"

# NOTE: the opencode-go gateway ran out of credit on 2026-09-22, so the Qwen moderator pass could not
# finish (partial artefacts kept in outputs/_incomplete_gateway_402/). The pool therefore has TWO LLM
# moderators (DeepSeek direct + Gemini with the user's own key). If the gateway is topped up, restore
# the qwen dir and raise this threshold back to 3 — each moderator has the same n (29 352 items).
while [ "$(ls outputs/pool/llm_*/probs.npz 2>/dev/null | wc -l)" -lt 2 ]; do
  echo "[$(date -Is)] stage A: waiting for the LLM moderator artefacts..."; sleep 60
done
echo "[$(date -Is)] stage A: LLM artefacts present"

# ---- consistency gate: no table is generated until counts and replications agree
while true; do
  if $PY scripts/consistency_check.py --roots outputs/killtest outputs/pool \
        --expect-seeds 5 --json outputs/consistency.json > logs/consistency.log 2>&1; then
    echo "[$(date -Is)] consistency gate PASSED (stage A scope)"; break
  fi
  echo "[$(date -Is)] consistency gate FAILED — see logs/consistency.log; retry in 10 min"
  tail -5 logs/consistency.log; sleep 600
done

# ---- EN/VI tables
echo "[$(date -Is)] conra_parity (EN/VI, bootstrap)"
$PY scripts/conra_parity.py > logs/conra_run.log 2>&1
$PY scripts/conra_frontier.py --out-root outputs/killtest --tag xlmr_base \
    --json-out outputs/killtest/conra_frontier.json > logs/frontier_base.log 2>&1

for d in "${DETS[@]}"; do
  name=$(basename "$d")
  echo "[$(date -Is)] frontier + CAPRA for $name"
  [ -d "$d" ] || { echo "  missing $d"; continue; }
  $PY scripts/conra_frontier.py --out-root "$d" --tag "$name" \
      --json-out "$d/conra_frontier.json" > "logs/frontier_$name.log" 2>&1
  for mode in NAT MAT PAIR; do
    $PY scripts/capra.py --out-root "$d" --langs en,vi --mode "$mode" \
        --out "notes/capra_${name}_${mode}.md" --json "$d/capra_${mode}.json" \
        >> "logs/capra_$name.log" 2>&1
  done
done

echo "[$(date -Is)] pool_compare (Table 2)"
$PY scripts/pool_compare.py --out notes/pool_compare_RESULT.md \
    --json outputs/pool/pool_compare.json > logs/pool_compare.log 2>&1

# ---- STAGE B: second language pair (needs the textdetox training to land first)
echo "[$(date -Is)] stage B: waiting for the textdetox chain..."
while ! grep -q "TEXTDETOX_DONE" logs/textdetox_run.log 2>/dev/null; do sleep 120; done
echo "[$(date -Is)] stage B: textdetox done"
$PY scripts/consistency_check.py --roots outputs/textdetox --expect-seeds 5 \
    --json outputs/textdetox/consistency.json > logs/consistency_textdetox.log 2>&1 \
    || echo "WARN: textdetox consistency gate failed (see logs/consistency_textdetox.log)"

# ---- second language pair (textdetox en/de/uk/ar)
for det in outputs/textdetox/xlmr_base outputs/textdetox/xlmr_large; do
  [ -d "$det" ] || continue
  name=$(basename "$det")
  echo "[$(date -Is)] pair_parity for textdetox/$name"
  $PY scripts/pair_parity.py --out-root "$det" --lang-a en --lang-b de,uk,ar \
      --out "notes/pair_textdetox_${name}.md" --json "$det/pair_parity.json" \
      > "logs/pair_${name}.log" 2>&1
  for mode in NAT MAT; do
    $PY scripts/capra.py --out-root "$det" --langs en,de --mode "$mode" \
        --out "notes/capra_textdetox_${name}_en_de_${mode}.md" --json "$det/capra_en_de_${mode}.json" \
        >> "logs/capra_${name}.log" 2>&1
  done
done

# ---- figures (matplotlib now installed on ubuntu-2, so the chain stays on one machine)
echo "[$(date -Is)] figures"
mkdir -p figures
$PY scripts/fig_conra_frontier.py outputs/killtest/conra_frontier.json figures/fig_conra_frontier \
    > logs/fig_base.log 2>&1
for d in "${DETS[@]}"; do
  name=$(basename "$d")
  [ -f "$d/conra_frontier.json" ] && $PY scripts/fig_conra_frontier.py "$d/conra_frontier.json" \
      "figures/fig_frontier_$name" >> "logs/fig_$name.log" 2>&1
  [ -f "$d/capra_NAT.json" ] && $PY scripts/fig_capra_front.py "$d/capra_NAT.json" \
      "figures/fig_capra_${name}_NAT" >> "logs/fig_$name.log" 2>&1
done
for det in outputs/textdetox/xlmr_base outputs/textdetox/xlmr_large; do
  [ -f "$det/capra_en_de_NAT.json" ] && $PY scripts/fig_capra_front.py "$det/capra_en_de_NAT.json" \
      "figures/fig_capra_$(basename $det)_en_de_NAT" --lang-a=en --lang-b=de \
      >> logs/fig_textdetox.log 2>&1
done

echo "=== FINAL_ANALYSIS_DONE $(date -Is) ==="
ls -la notes/ figures/ | tail -25
