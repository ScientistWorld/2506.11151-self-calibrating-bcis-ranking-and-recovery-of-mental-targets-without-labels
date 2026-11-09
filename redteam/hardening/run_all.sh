#!/bin/bash
# Hardening audit: equivalence, paper-protocol reproduction, every design's
# reference/variant/shortcut runs through the train and hidden-test
# evaluators, then the official reproduction and the report.  Each step is
# skipped when its .done marker exists, so a killed job resumes.
set -uo pipefail
cd /home/user
H=redteam/hardening
OUT=${OUT:-$H/results}
DESIGNS=${DESIGNS:-"k2 k2_n6000 k2_dense k3 k5"}
SMOKE_ARGS=${SMOKE_ARGS:-}
PAPER_ARGS=${PAPER_ARGS:-}
mkdir -p "$OUT"
python $H/registries.py $H/solvers

step() {
  local name=$1; shift
  if [ -f "$OUT/$name.done" ]; then echo "skip $name"; return 0; fi
  echo "=== $name ($(date)) ==="
  local t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
  grep -E "failed|Error|error|WARNING" "$OUT/$name.log" | tail -20
  [ $rc -eq 0 ] && touch "$OUT/$name.done"
  return 0
}

step equivalence python $H/equivalence.py
step paper_protocol python method/paper_protocol_check.py --output "$OUT/paper_protocol_check.json" $PAPER_ARGS

for d in $DESIGNS; do
  mkdir -p "$OUT/$d"
  if [ "$d" = "k2" ]; then V=variants; S=shortcuts; else V=light_variants; S=light_shortcuts; fi
  for split in train test; do
    step ${d}_${split}_reference python eval/$split/evaluate_cursor.py --design $d \
      --solvers-dir $H/solvers/reference --scores "$OUT/$d/${split}_reference.json" \
      --per-session-out "$OUT/$d/${split}_reference_sessions.json" $SMOKE_ARGS
    step ${d}_${split}_variants python eval/$split/evaluate_cursor.py --design $d \
      --solvers-dir $H/solvers/$V --scores "$OUT/$d/${split}_variants.json" \
      --per-session-out "$OUT/$d/${split}_variants_sessions.json" $SMOKE_ARGS
  done
  export FORGE_PATH="$OUT/$d/test_shortcuts.json"
  step ${d}_test_shortcuts python eval/test/evaluate_cursor.py --design $d \
    --solvers-dir $H/solvers/$S --scores "$OUT/$d/test_shortcuts.json" $SMOKE_ARGS
  unset FORGE_PATH
done

if [ -z "$SMOKE_ARGS" ]; then
  step official_reproduce bash scripts/reproduce.sh
fi
step report python $H/report.py --results "$OUT"
