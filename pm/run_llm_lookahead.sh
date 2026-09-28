#!/usr/bin/env bash
# Score every model in pm/llm_lookahead.py's roster, one at a time, under a memory
# watchdog: the lum.id sandbox's cgroup is 8 GB and an OOM kills the whole pod,
# taking the concurrent RL sweep with it (SANDBOX.md), so any scorer whose
# cgroup's anonymous memory passes the ceiling is killed first. Resumable: a model whose
# parquet exists is skipped.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PYTHON="${PYTHON:-/opt/venv/bin/python}"
# hf_xet buffers download chunks in anonymous memory (~3 GB seen), enough to
# push the cgroup over 8 GB beside one RL worker; plain HTTP streams to disk.
export HF_HOME="${HF_HOME:-/root/.cache/hf}" HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
CEIL_MB="${CEIL_MB:-6800}"            # anon memory of the whole cgroup
mkdir -p pm/logs
ROSTER="${ROSTER:-unsloth/Meta-Llama-3.1-8B-Instruct:none
unsloth/gemma-3-12b-it:8bit
allenai/Olmo-3-7B-Instruct:none
google/gemma-4-12B-it:8bit
mistralai/Ministral-3-8B-Instruct-2512-BF16:none
Qwen/Qwen3.5-9B:none}"
PROMPT="${PROMPT:-forecast}"
while IFS=: read -r model quant; do
  [ -z "$model" ] && continue
  log="pm/logs/llm_${model//\//__}_${PROMPT}.log"
  "$PYTHON" pm/llm_lookahead.py --model "$model" --quant "$quant" --prompt "$PROMPT" > "$log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    anon=$(( $(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat) / 1048576 ))
    if [ "$anon" -gt "$CEIL_MB" ]; then
      echo "KILL $model: cgroup anon ${anon} MB > ${CEIL_MB}" | tee -a "$log"; kill -9 $pid
    fi
    sleep 2
  done
  wait $pid; echo "$model exit $? $(date +%H:%M)"; tail -1 "$log"
done <<< "$ROSTER"
echo LLM ROSTER COMPLETE
