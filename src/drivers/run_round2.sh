#!/usr/bin/env bash
# Round 2 — chained, unattended. Everything on ubuntu-2 (standing rule: no Mac/remote split).
#
#   stage 0  prepare textdetox with 8 languages, every language capped to the SAME n
#   stage 1  GPU: shared detectors xlmr_base + xlmr_large, 5 seeds each (background)
#   stage 2  LLM moderators over the same 8 languages (network-bound, runs in parallel with stage 1)
#   stage 3  pair analysis for en<->{de,uk,ru,ar,he,hi,zh} x every detector
#   stage 4  CAPRA fronts for the trained detectors on the most interesting targets
#   stage 5  prior-art baselines (group-conditional risk control, min-max, equalized conformal)
#   stage 6  ablations (composition target, review cost, traffic mix, calibration size, seed spread)
#   stage 7  figures (capacity by language + CAPRA fronts) and the consistency gate
#   stage 8  regenerate the digest; print R2_DONE
set -u
cd .
PY=$HOME/miniconda3/envs/toxkgvi-gpu-ft/bin/python
DA=data/textdetox
ROOT=outputs/textdetox
# Hebrew is EXCLUDED: the dataset ships only 2 011 items for it (every other language has 5 000), so it
# cannot meet the equal-n convention that binds every table in this project. Documented, not silent.
LANGS=en,de,uk,ru,ar,hi,zh
TARGETS=de,uk,ru,ar,hi,zh
SEEDS=(42 43 44 45 46)
mkdir -p logs

echo "=== R2 START $(date -Is) ==="

# ---------------------------------------------------------------- stage 0
echo "--- stage 0: prep 8 languages, common n"
$PY scripts/textdetox_prep.py --out "$DA" --langs "$LANGS" --common-n 4500 > logs/r2_prep.log 2>&1 \
  || { echo "PREP FAILED"; tail -5 logs/r2_prep.log; exit 1; }
tail -9 logs/r2_prep.log
# guard: a language that could not reach the common n would break the equal-n convention downstream
$PY - <<'PYEOF' || { echo "PREP GUARD FAILED (short language)"; exit 1; }
import json, sys
s = json.load(open("data/textdetox/summary.json"))
n = s["args"]["common_n"]
short = {k: v["used"] for k, v in s["langs"].items() if n and v["used"] < n}
if short:
    print("SHORT LANGUAGES:", short, file=sys.stderr)
    raise SystemExit(1)
print("prep guard OK: every language has", n, "items")
PYEOF

# the 4-language artefacts are superseded: archive instead of mixing two generations of data
if [ -d "$ROOT" ]; then
  mv "$ROOT" outputs/_superseded_textdetox4lang_$(date +%H%M) || true
fi
mkdir -p "$ROOT"

# ---------------------------------------------------------------- stage 1 (GPU, background)
(
  for M in base large; do
    for S in "${SEEDS[@]}"; do
      if [ "$M" = "base" ]; then MODEL=xlm-roberta-base; BS=32; GA=1; else MODEL=FacebookAI/xlm-roberta-large; BS=8; GA=4; fi
      echo "--- $M seed$S $(date -Is)"
      $PY scripts/pool_train.py --model "$MODEL" --langs "$LANGS" --data-dir "$DA" \
          --bs "$BS" --grad-accum "$GA" --seed "$S" --out "$ROOT/xlmr_$M/seed$S" \
          || echo "FAILED $M seed$S"
      rm -rf "$ROOT/xlmr_$M/seed$S/ckpt"
    done
  done
  echo "R2_TRAIN_DONE $(date -Is)"
  ls "$ROOT"/xlmr_*/seed*/probs.npz 2>/dev/null | wc -l
) > logs/r2_train.log 2>&1 &
TRAIN_PID=$!
echo "training started (pid $TRAIN_PID)"

# ---------------------------------------------------------------- stage 2 (network, parallel)
set -a; . "$HOME/.hermes_keys.env"; set +a
DEVS=$(echo "$LANGS" | tr ',' '\n' | sed 's/$/_dev/' | paste -sd, -)
TESTS=$(echo "$LANGS" | tr ',' '\n' | sed 's/$/_test/' | paste -sd, -)
SETS="$DEVS,$TESTS"
echo "--- stage 2: LLM moderators on $SETS"
$PY scripts/pool_llm.py --model deepseek-flash --base-url https://api.deepseek.com \
    --key-env DEEPSEEK_API_KEY --thinking disabled --data-dir "$DA" --sets "$SETS" \
    --workers 12 --max-tokens 200 --out "$ROOT/llm_deepseek_flash" \
    > logs/r2_llm_deepseek.log 2>&1 || echo "FAILED deepseek"
$PY scripts/pool_llm.py --model gemini-3.1-flash-lite \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai \
    --key-env GEMINI_API_KEY --no-json-mode --data-dir "$DA" --sets "$SETS" \
    --workers 8 --max-tokens 200 --out "$ROOT/llm_gemini31_flashlite" \
    > logs/r2_llm_gemini.log 2>&1 || echo "FAILED gemini"
$PY scripts/pool_llm.py --model qwen3.8-flash --base-url https://opencode.ai/zen/go/v1 \
    --key-env OPENCODE_GO_API_KEY --opencode --data-dir "$DA" --sets "$SETS" \
    --workers 24 --max-tokens 200 --out "$ROOT/llm_qwen38_flash" \
    > logs/r2_llm_qwen.log 2>&1 || echo "FAILED qwen (gateway credit?)"

# ---------------------------------------------------------------- wait for the GPU
echo "--- waiting for training"
wait "$TRAIN_PID"
grep -a "R2_TRAIN_DONE" logs/r2_train.log || echo "WARN: training marker missing"

# ---------------------------------------------------------------- stage 3-6 (CPU)
echo "--- stage 3: pair analysis"
for det in xlmr_base xlmr_large llm_deepseek_flash llm_gemini31_flashlite llm_qwen38_flash; do
  [ -d "$ROOT/$det" ] || continue
  # a moderator with unscored items would break the equal-n convention: check before analysing
  bad=$($PY - "$ROOT/$det" <<'PYEOF'
import json, os, sys
p = os.path.join(sys.argv[1], "usage.json")
if not os.path.exists(p):
    print("no-usage-json"); raise SystemExit
c = json.load(open(p)).get("counts", {})
bad = [k for k, v in c.items() if v.get("defaulted")]
print(",".join(bad) if bad else "")
PYEOF
)
  if [ -n "$bad" ]; then echo "  [skip] $det has unscored items: $bad"; continue; fi
  $PY scripts/pair_parity.py --out-root "$ROOT/$det" --lang-a en --lang-b "$TARGETS" \
      --out "notes/pair_textdetox_$det.md" --json "$ROOT/$det/pair_parity.json" \
      >> logs/r2_analysis.log 2>&1 || echo "FAILED pair $det"
  echo "  pair $det done"
done

echo "--- stage 4: CAPRA fronts"
for det in xlmr_base xlmr_large; do
  [ -d "$ROOT/$det" ] || continue
  for t in de ar zh; do
    $PY scripts/capra.py --out-root "$ROOT/$det" --langs en,$t --mode NAT \
        --out "notes/capra_textdetox_${det}_en_${t}_NAT.md" --json "$ROOT/$det/capra_en_${t}_NAT.json" \
        >> logs/r2_analysis.log 2>&1 || echo "FAILED capra $det/$t"
    $PY scripts/fig_capra_front.py "$ROOT/$det/capra_en_${t}_NAT.json" \
        "figures/fig_capra_textdetox_${det}_en_${t}" >> logs/r2_analysis.log 2>&1 || true
  done
done

echo "--- stage 5: prior-art baselines"
$PY scripts/baseline_rules.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --budgets 0.05 --out notes/baseline_killtest_NAT.md --json outputs/killtest/baseline_NAT.json \
    >> logs/r2_analysis.log 2>&1 || echo "FAILED baseline killtest NAT"
for det in xlmr_large specialist_pair llm_deepseek_flash llm_gemini31_flashlite llm_qwen38_flash; do
  [ -d "outputs/pool/$det" ] || continue
  $PY scripts/baseline_rules.py --out-root "outputs/pool/$det" --lang-a en --lang-b vi --mode NAT \
      --budgets 0.05 --out "notes/baseline_${det}_NAT.md" --json "outputs/pool/$det/baseline_NAT.json" \
      >> logs/r2_analysis.log 2>&1 || echo "FAILED baseline $det"
done
for det in xlmr_base xlmr_large; do
  $PY scripts/baseline_rules.py --out-root "$ROOT/$det" --lang-a en --lang-b "$TARGETS" \
      --budgets 0.05 --out "notes/baseline_textdetox_$det.md" --json "$ROOT/$det/baseline.json" \
      >> logs/r2_analysis.log 2>&1 || echo "FAILED baseline textdetox $det"
done

echo "--- stage 6: ablations"
$PY scripts/ablate.py --out-root outputs/killtest --lang-a en --lang-b vi --mode NAT \
    --out notes/ablations_killtest_NAT.md --json outputs/killtest/ablations_NAT.json \
    >> logs/r2_analysis.log 2>&1 || echo "FAILED ablate killtest"
for det in xlmr_base xlmr_large; do
  $PY scripts/ablate.py --out-root "$ROOT/$det" --lang-a en --lang-b "$TARGETS" \
      --out "notes/ablations_textdetox_$det.md" --json "$ROOT/$det/ablations.json" \
      >> logs/r2_analysis.log 2>&1 || echo "FAILED ablate $det"
done

# ---------------------------------------------------------------- stage 7
echo "--- stage 7: figures + gate"
$PY scripts/fig_capacity_by_lang.py --root "$ROOT" --out figures/fig_capacity_by_lang \
    >> logs/r2_analysis.log 2>&1 || echo "FAILED fig capacity"
$PY scripts/consistency_check.py outputs/killtest outputs/pool outputs/textdetox \
    > logs/r2_gate.log 2>&1 && echo "gate OK" || echo "GATE FAILED (see logs/r2_gate.log)"

# ---------------------------------------------------------------- stage 8
echo "--- stage 8: digest"
$PY scripts/final_summary.py --out notes/FINAL_SUMMARY.md >> logs/r2_analysis.log 2>&1 || echo "FAILED digest"
echo "=== R2_DONE $(date -Is) ==="
tail -3 logs/r2_gate.log 2>/dev/null
