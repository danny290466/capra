#!/usr/bin/env bash
# Round-1 review remediation: re-run every affected analysis from the FROZEN artefacts (no retraining,
# no new data), now that the four implementation bugs are fixed:
#   * price of composition computed on one split (test - test)
#   * min_capacity type bug (filled a NaN column)
#   * cluster bootstrap over unique test items (was 5x pseudo-replicated)
#   * Student t for the seed-level test (was a normal approximation at n = 5)
# plus the new analyses the reviewers asked for: null policy + test-optimal oracle at matched load,
# realised test load per rule, the three operating modes, the prevalence-matched control, and
# significance for the hosted moderators and the second benchmark.
set -u
cd .
PY=$HOME/miniconda3/envs/toxkgvi-gpu-ft/bin/python
TARGETS=de,uk,ru,ar,hi,zh

echo "=== R3 REMEDIATION START $(date -Is) ==="

echo "--- 1. CAPRA fronts (test-minus-test prices)"
$PY scripts/capra.py --out-root outputs/killtest --langs en,vi --mode NAT \
    --budgets 0.05,0.10,0.20 --out notes/capra_RESULT.md --json outputs/killtest/capra.json >/dev/null 2>&1 \
    && echo "  capra killtest ok"
for det in xlmr_base xlmr_large; do
  for t in de ar zh; do
    $PY scripts/capra.py --out-root outputs/textdetox/$det --langs en,$t --mode NAT \
        --out notes/capra_textdetox_${det}_en_${t}_NAT.md \
        --json outputs/textdetox/$det/capra_en_${t}_NAT.json >> logs/r3_analysis.log 2>&1
  done
  $PY scripts/fig_capra_front.py outputs/textdetox/$det/capra_en_de_NAT.json \
      figures/fig_capra_textdetox_${det}_en_de >> logs/r3_analysis.log 2>&1
done
echo "  capra textdetox ok"

echo "--- 2. ablations (capacity column filled)"
$PY scripts/ablate.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --out notes/ablations_killtest_NAT.md --json outputs/killtest/ablations_NAT.json >> logs/r3_analysis.log 2>&1 \
    && echo "  ablate killtest ok"
for det in xlmr_base xlmr_large; do
  $PY scripts/ablate.py --out-root outputs/textdetox/$det --lang-a en --lang-b $TARGETS \
      --out notes/ablations_textdetox_$det.md --json outputs/textdetox/$det/ablations.json >> logs/r3_analysis.log 2>&1 \
      && echo "  ablate $det ok"
done

echo "--- 3. prior-art audit (null + oracle + realised test load)"
$PY scripts/baseline_rules.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --budgets 0.05 --out notes/baseline_killtest_NAT.md --json outputs/killtest/baseline_NAT.json >/dev/null 2>&1 \
    && echo "  baseline killtest NAT ok"
for det in xlmr_large specialist_pair llm_deepseek_flash llm_gemini31_flashlite llm_qwen38_flash; do
  [ -d outputs/pool/$det ] || continue
  $PY scripts/baseline_rules.py --out-root outputs/pool/$det --lang-a en --lang-b vi --mode NAT \
      --budgets 0.05 --out notes/baseline_${det}_NAT.md --json outputs/pool/$det/baseline_NAT.json >> logs/r3_analysis.log 2>&1 \
      && echo "  baseline $det ok"
done
for det in xlmr_base xlmr_large; do
  $PY scripts/baseline_rules.py --out-root outputs/textdetox/$det --lang-a en --lang-b $TARGETS \
      --budgets 0.05 --out notes/baseline_textdetox_$det.md --json outputs/textdetox/$det/baseline.json >> logs/r3_analysis.log 2>&1 \
      && echo "  baseline textdetox $det ok"
done

echo "--- 4. significance (cluster bootstrap + Student t) for every detector"
$PY scripts/significance.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --budgets 0.05 --out notes/significance_killtest_NAT.md --json outputs/killtest/significance_NAT.json >/dev/null 2>&1 \
    && echo "  significance killtest ok"
for det in llm_deepseek_flash llm_gemini31_flashlite llm_qwen38_flash; do
  [ -d outputs/pool/$det ] || continue
  $PY scripts/significance.py --out-root outputs/pool/$det --lang-a en --lang-b vi --mode NAT \
      --budgets 0.05 --out notes/significance_${det}_NAT.md --json outputs/pool/$det/significance_NAT.json \
      >> logs/r3_analysis.log 2>&1 && echo "  significance $det ok"
done
for det in xlmr_base xlmr_large; do
  $PY scripts/significance.py --out-root outputs/textdetox/$det --lang-a en --lang-b $TARGETS \
      --budgets 0.05 --out notes/significance_textdetox_$det.md --json outputs/textdetox/$det/significance.json \
      >> logs/r3_analysis.log 2>&1 && echo "  significance textdetox $det ok"
done

echo "--- 5. the three operating modes (NAT / MAT / PAIR)"
for mode in NAT MAT PAIR; do
  $PY scripts/conra_parity.py --out-root outputs/killtest --mode $mode \
      --out notes/conra_parity_${mode}_RESULT.md --json outputs/killtest/conra_parity_${mode}.json \
      >> logs/r3_analysis.log 2>&1 && echo "  mode $mode ok"
done
# prevalence-matched control for the second benchmark
for det in xlmr_base xlmr_large; do
  for t in de ar; do
    $PY scripts/baseline_rules.py --out-root outputs/textdetox/$det --lang-a en --lang-b $t \
        --budgets 0.05 --match-prevalence 0.5 \
        --out notes/baseline_textdetox_${det}_en_${t}_prevmatched.md \
        --json outputs/textdetox/$det/baseline_prevmatched_en_${t}.json >> logs/r3_analysis.log 2>&1
  done
done
echo "  prevalence-matched control done"

echo "--- 6. gate, stats, digest"
$PY scripts/consistency_check.py --roots outputs/killtest outputs/pool outputs/textdetox \
    --expect-seeds 5 --groups "envi=outputs/killtest,outputs/pool" "textdetox=outputs/textdetox" \
    --json outputs/consistency.json > logs/r3_gate.log 2>&1 && echo "  gate OK" || echo "  GATE FAILED"
$PY scripts/load_stats.py --proj . --out manuscript/stats.json >/dev/null 2>&1 && echo "  stats ok"
$PY scripts/final_summary.py --out notes/FINAL_SUMMARY.md >/dev/null 2>&1 && echo "  digest ok"

echo "=== R3_REMEDIATION_DONE $(date -Is) ==="
