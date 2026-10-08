#!/bin/bash
# Start the GeoEG demo page: launches vLLM (scripts/start_vllm.sh) if it is not running, then the web backend.
# Usage (inside WSL Ubuntu-22.04): bash web/start_web.sh      then open http://localhost:7860 in Windows
# Stop: Ctrl+C stops the web page; bash scripts/stop_vllm.sh frees the ~10.5GB VRAM of the model.
set -u
cd "$(dirname "$0")"
PORT=${PORT:-7860}

healthy() { curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null; }

# Seen once: vLLM listening on 8001 but every loopback connect refused (WSL mirrored networking).
# A fresh restart fixed it, so retry once with stop + start.
for attempt in 1 2; do
  healthy && break
  echo "Starting vLLM (about 1 minute)..."
  bash ../scripts/stop_vllm.sh >/dev/null
  bash ../scripts/start_vllm.sh
done
healthy || { echo "vLLM is not reachable on 127.0.0.1:8001; see ~/vllm_serve.log"; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geoeg   # GeoEG pipeline is imported in-process; the 7B model stays in vLLM
echo "Open http://localhost:$PORT in your browser (Ctrl+C to stop the page)"
exec uvicorn server:app --host 0.0.0.0 --port "$PORT"
