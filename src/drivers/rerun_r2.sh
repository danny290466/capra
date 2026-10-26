#!/usr/bin/env bash
# rerun_r2.sh — Round-2 remediation driver. Re-derives EVERY affected analysis from the FROZEN artefacts:
# no retraining, no new data, CPU only. Run from the project root on ubuntu-2.
#
# Fixes driven by the code audit (R3) and the two Round-2 reviews:
#   R3-01/R2-14  the gate's seed check was keyed on the substring "seed42" and never fired
#   R3-03/R2-02  the printed price of composition was a design-split difference
#   R3-04/R2-18  the ablation capacity column raised KeyError and printed NaN in every cell
#   R3-05/R3-06  unscored items were written as p = 0.5 (minimum confidence) instead of NaN
#   R3-08        the "test-optimal oracle" refitted its thresholds on calibration and equalled CAPRA
#   R3-09        the operating-modes table had no input file at the path the build reads
#   R3-10        metrics.json lives in each seed directory, not at the detector root
#   R3-14/R3-15  the null policy was not simulated; MINMAX-CAP was EQUAL-COV under another name
#   R3-18        the seed spread used |D| where the signed gap was meant
#   R3-22        only one detector had a capra.json, so five rows of Table 4 used another rule's number
#   R3-23        every driver called scripts/<module>.py while the modules sit at the top level
set -uo pipefail
cd .
PY=~/miniconda3/envs/toxkgvi-gpu-ft/bin/python
mkdir -p logs
LOG=logs/rerun_r2.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] RERUN_R2_START"
ENVI_DETS="xlmr_base:outputs/killtest xlmr_large:outputs/pool/xlmr_large specialist_pair:outputs/pool/specialist_pair llm_deepseek_flash:outputs/pool/llm_deepseek_flash llm_gemini31_flashlite:outputs/pool/llm_gemini31_flashlite"
TARGETS="de,uk,ru,ar,hi,zh"

echo "[$(date -Is)] 0/7 detector pool (Table 4, without the detector that left items unscored)"
$PY pool_compare.py --out notes/pool_compare_RESULT.md --json outputs/pool/pool_compare.json \
    > logs/pool_compare_r2.log 2>&1 || echo "  WARN pool_compare"

echo "[$(date -Is)] 1/7 CAPRA endpoints for every EN/VI detector"
for spec in $ENVI_DETS; do
  name=${spec%%:*}; d=${spec#*:}
  $PY capra.py --out-root "$d" --langs en,vi --mode NAT \
      --out "notes/capra_${name}_NAT.md" --json "$d/capra.json" \
      > "logs/capra_${name}_r2.log" 2>&1 || echo "  WARN capra $name"
done

echo "[$(date -Is)] 2/7 prior-art audit (Table 9)"
$PY baseline_rules.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --out notes/baseline_killtest_NAT.md --json outputs/killtest/baseline_NAT.json \
    > logs/baseline_envi_r2.log 2>&1 || echo "  WARN baseline envi"
for det in xlmr_base xlmr_large; do
  $PY baseline_rules.py --out-root "outputs/textdetox/$det" --lang-a en --lang-b "$TARGETS" --mode NAT \
      --budgets 0.05,0.10,0.20 --out "notes/baseline_textdetox_${det}.md" \
      --json "outputs/textdetox/$det/baseline.json" \
      > "logs/baseline_${det}_r2.log" 2>&1 || echo "  WARN baseline $det"
done

echo "[$(date -Is)] 3/7 ablations (Table 7 / Appendix B)"
$PY ablate.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --out notes/ablations_killtest_NAT.md --json outputs/killtest/ablations_NAT.json \
    > logs/ablate_envi_r2.log 2>&1 || echo "  WARN ablate envi"
for det in xlmr_base xlmr_large; do
  $PY ablate.py --out-root "outputs/textdetox/$det" --lang-a en --lang-b "$TARGETS" --mode NAT \
      --out "notes/ablations_textdetox_${det}.md" --json "outputs/textdetox/$det/ablations.json" \
      > "logs/ablate_${det}_r2.log" 2>&1 || echo "  WARN ablate $det"
done

echo "[$(date -Is)] 4/7 significance (Table 10): signed gap, diagnostics, seed-level dagger"
$PY significance.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --out notes/significance_killtest_NAT.md --json outputs/killtest/significance_NAT.json \
    > logs/sig_envi_r2.log 2>&1 || echo "  WARN sig envi"
for det in llm_deepseek_flash llm_gemini31_flashlite; do
  $PY significance.py --out-root "outputs/pool/$det" --lang-a en --lang-b vi --mode NAT \
      --out "notes/significance_${det}_NAT.md" --json "outputs/pool/$det/significance_NAT.json" \
      > "logs/sig_${det}_r2.log" 2>&1 || echo "  WARN sig $det"
done
for det in xlmr_base xlmr_large; do
  $PY significance.py --out-root "outputs/textdetox/$det" --lang-a en --lang-b "$TARGETS" --mode NAT \
      --out "notes/significance_textdetox_${det}.md" --json "outputs/textdetox/$det/significance.json" \
      > "logs/sig_td_${det}_r2.log" 2>&1 || echo "  WARN sig td $det"
done

echo "[$(date -Is)] 5/7 operating modes (Table 7)"
$PY modes_report.py --out notes/modes_ENVI.md --json outputs/killtest/modes_ENVI.json \
    > logs/modes_r2.log 2>&1 || echo "  WARN modes"

echo "[$(date -Is)] 6/7 consistency gate (now enforcing the seed count and unscored items)"
$PY consistency_check.py --roots outputs/killtest outputs/pool outputs/textdetox \
    --expect-seeds 5 --groups "envi=outputs/killtest,outputs/pool" "textdetox=outputs/textdetox" \
    --json outputs/consistency.json > logs/consistency_r2.log 2>&1 || echo "  GATE_FAILED (see logs/consistency_r2.log)"

echo "[$(date -Is)] 7/7 bundle + digest"
$PY scripts/load_stats.py --proj . --out manuscript/stats.json > logs/load_stats_r2.log 2>&1 \
    && echo "  stats ok" || echo "  WARN load_stats"
$PY - <<'EOF'
import json, hashlib
p = "manuscript/stats.json"
raw = open(p, "rb").read()
d = json.load(open(p))
json.dump({"sha256_16": hashlib.sha256(raw).hexdigest()[:16], "bytes": len(raw),
           "digest": d.get("digest")}, open("manuscript/stats_digest.json", "w"), indent=2)
print("  digest", hashlib.sha256(raw).hexdigest()[:16], len(raw), "bytes")
EOF
echo "[$(date -Is)] RERUN_R2_DONE"
