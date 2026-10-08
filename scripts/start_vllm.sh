#!/bin/bash
# Start the Qwen2.5-VL-7B-Instruct-AWQ OpenAI-compatible server (vLLM, conda env geoeg-vllm) on 127.0.0.1:8001.
# Usage (inside WSL Ubuntu-22.04): bash start_vllm.sh [extra vllm args...]   log: ~/vllm_serve.log
# Port 8000 is NOT used: vLLM listens there but loopback connections get refused (cause not investigated yet).
set -u
PORT=8001
LOG=~/vllm_serve.log

if pgrep -f 'bin/vllm serve' >/dev/null; then
  echo "vLLM already running (pid $(pgrep -f 'bin/vllm serve' | head -1)); stop it first with stop_vllm.sh"
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoeg-vllm
export MAX_JOBS=4
# Windows interop dirs (e.g. CUDA 11.8 nvcc under /mnt/c) break torch.compile; keep Linux paths only
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
# no system CUDA toolkit in WSL: avoid flashinfer's JIT-compiled sampler
export VLLM_USE_FLASHINFER_SAMPLER=0

# 1003520 = 1280*28*28 pixels max per image; 200704 = 256*28*28 min
nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name qwen2.5-vl-7b-awq \
  --gpu-memory-utilization 0.6 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"min_pixels": 200704, "max_pixels": 1003520}' \
  "$@" > "$LOG" 2>&1 &

# health check: startup takes ~60s (torch.compile + CUDA graphs)
for i in $(seq 1 60); do
  sleep 5
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "vLLM ready on http://127.0.0.1:$PORT/v1 after $((i * 5))s (model: qwen2.5-vl-7b-awq)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  if ! pgrep -f 'bin/vllm serve' >/dev/null; then
    echo "vLLM exited during startup; last log lines:"
    tail -n 20 "$LOG"
    exit 1
  fi
done
echo "vLLM not healthy after 300s; see $LOG"
exit 1
