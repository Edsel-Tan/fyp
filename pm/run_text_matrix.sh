#!/usr/bin/env bash
# Full experimental matrix for the prediction-market text study (RESULTS.md sec.3.6).
#
# Resumable: every arm writes data/nlp/signal_<tag>.json and an arm whose JSON
# already exists is skipped, so the matrix can be interrupted and restarted
# without losing completed work. Delete the JSON to force a rerun.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=data/nlp/matrix.log
F=8
mkdir -p data/nlp/arms
touch "$LOG"

# Mirror pmv1_text_signal.main()'s default tag so completed arms can be detected.
tag_of() {
  local text=tfidf field=text kind=nl target=term split=walk null=none resid=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --text) text=$2; shift 2;; --field) field=$2; shift 2;;
      --kind) kind=$2; shift 2;; --target) target=$2; shift 2;;
      --split) split=$2; shift 2;; --null) null=$2; shift 2;;
      --residual) resid="_resid"; shift;; *) shift;;
    esac
  done
  echo "${text}_${field}_${kind}_${target}_${split}_${null}${resid}"
}

run() {
  local tag; tag=$(tag_of "$@")
  if [ -f "data/nlp/signal_${tag}.json" ]; then
    echo "--- skip (done): $*" >> "$LOG"; return
  fi
  echo "### $*" >> "$LOG"
  $PY pm/pmv1_text_signal.py "$@" > "data/nlp/arms/${tag}.out" 2>&1 || \
    echo "  ARM FAILED -- see data/nlp/arms/${tag}.out" >> "$LOG"
  grep -E "^panel|^ *fold |^pooled|IC\(M|^ +dIC|book:" "data/nlp/arms/${tag}.out" >> "$LOG"
  echo >> "$LOG"
}

# A. which encoder, if any, reads the question -- honest walk-forward split
for T in tfidf finbert bert minilm; do run --text $T --kind nl --folds $F; done
# B. does masking the calendar out of the string change the answer
for T in tfidf finbert; do run --text $T --kind nl --field text_scrub --folds $F; done
# C. the leak, decomposed: random rows vs random events vs forward in time
for S in random eventrandom; do
  run --text tfidf   --kind all --split $S
  run --text finbert --kind all --split $S
done
run --text tfidf   --kind all --folds $F
run --text finbert --kind all --folds $F
# D. falsification: destroy the language, hold category and price level
for T in tfidf finbert; do run --text $T --kind nl --null shuffle --folds $F; done
# E. text shown only what the price model could not explain, and booked alone
for T in tfidf finbert; do run --text $T --kind nl --residual --folds $F; done
# F. the machine-templated stratum, where the "text" is a clock
for T in tfidf finbert; do run --text $T --kind date --folds $F; done
# G. the other target: 24h forward drift rather than hold-to-resolution
for T in tfidf finbert; do run --text $T --kind nl --target fwd --folds $F; done
echo "MATRIX DONE" >> "$LOG"
