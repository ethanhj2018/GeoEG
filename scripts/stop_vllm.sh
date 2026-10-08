#!/bin/bash
# Stop the vLLM server started by start_vllm.sh and show GPU memory afterwards.
# Usage (inside WSL Ubuntu-22.04): bash stop_vllm.sh
if ! pgrep -f 'bin/vllm serve' >/dev/null; then
  echo "vLLM is not running"
else
  pkill -f 'bin/vllm serve'
  for i in $(seq 1 30); do
    pgrep -f 'bin/vllm serve' >/dev/null || break
    sleep 1
  done
  if pgrep -f 'bin/vllm serve' >/dev/null; then
    echo "still running after 30s; sending SIGKILL"
    pkill -9 -f 'bin/vllm serve'
  fi
  sleep 3
  echo "vLLM stopped"
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
