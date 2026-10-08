"""Send one image to the local vLLM server (start_vllm.sh) and report latency, token counts and GPU memory.
Usage (conda env geoeg-vllm or any Python 3): python test_vllm_client.py <image> [--port 8001] [--runs 3] [--prompt "..."]
"""
import argparse, base64, json, mimetypes, subprocess, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--runs", type=int, default=3, help="later runs hit the prefix cache; run 0 is the cold number")
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--prompt", default="Read all text in this image, then guess which city/country it was taken in. Answer briefly.")
args = ap.parse_args()

def gpu_used():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip() + " MiB"
    except FileNotFoundError:
        return "n/a"

mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"
b64 = base64.b64encode(open(args.image, "rb").read()).decode()
body = {
    "model": "qwen2.5-vl-7b-awq",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": args.prompt},
    ]}],
    "max_tokens": args.max_tokens,
    "temperature": 0,
}
url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
for i in range(args.runs):
    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    dt = time.time() - t0
    u = resp["usage"]
    print(f"run {i}: {dt * 1000:.0f} ms, prompt_tokens={u['prompt_tokens']} "
          f"completion_tokens={u['completion_tokens']}, GPU used {gpu_used()}")
print("answer:", resp["choices"][0]["message"]["content"])
