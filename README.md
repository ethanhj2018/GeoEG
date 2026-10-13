# GeoEG: Geospatial Evidence Graph

GeoEG locates where a photo was taken. Given one image, it returns the Top-5 candidate locations, an evidence graph (JSON) and an explanation that cites each clue it relied on. It combines a vision-language model (VLM) with deterministic evidence (OCR text, place-name lookup, licence plates, calling codes, driving side) and image-retrieval models (GeoCLIP, G3), and it writes every intermediate result to disk so each prediction can be traced back.

## How it works

```
image ─► 1  in parallel: VLM direct coordinate prediction (the anchor)
                       ‖ VLM scene description (what is visible, no location verdicts)
                       ‖ PaddleOCR (persistent worker)
      ─► 2  isolate clues: Grounding DINO boxes → SAM 2 masks (text / object / infra clues only)
      ─► 3  deterministic evidence: writing system, calling codes, Chinese licence plates,
            place names found in text (Nominatim, disambiguated by co-occurring province/country),
            driving side
      ─► 4  candidate pool: VLM anchor ∪ place-name candidates ∪ GeoCLIP gallery Top-20,
            each scored with G3 + GeoCLIP image–location similarity
      ─► 5  decision (conclude/decide.py, parameters tuned on a dev split):
            · a clear, confirmed place name is close to decisive
            · switch away from the anchor only when retrieval clearly favours another candidate
            · refine within 200 km of the anchor by local G3 + GeoCLIP re-ranking
      ─► 6  evidence graph (clue and candidate nodes; support / refine / contradict edges)
            + Top-5 + explanation, saved under runs/<time>_<image>/
```

The key design choice is to keep **observation separate from conclusion**: the VLM describes what it sees, while location evidence comes only from deterministic sources. One wrong country guess therefore cannot spread across every clue.

## Repository layout

| Path | Contents |
|---|---|
| `geoeg/geoeg/schema.py` | Dataclasses for clue nodes, edges and candidates; country-name normalisation |
| `geoeg/geoeg/extract/` | VLM prompts (direct prediction and description), detection, segmentation, OCR client and worker |
| `geoeg/geoeg/tools/` | Text clues, GeoCLIP prior, G3 scorer, geocoding, geo-sleuth lookups, request cache |
| `geoeg/geoeg/graph/` | NetworkX evidence graph with automatic edges, saved as JSON |
| `geoeg/geoeg/conclude/` | Decision layer (pure functions, replayable offline) and candidate handling |
| `geoeg/geoeg/eval/` | Metrics, dataset loader, IM2GPS3K evaluation and tuning, `results/` |
| `geoeg/geoeg/pipeline.py` | End-to-end pipeline; `gpu.py` loads and frees models on demand |
| `geoeg/geoeg/cli.py`, `geoeg/app.py` | Command line and Gradio interface |
| `web/` | Demo web page (FastAPI backend + single-page front end, with example images) |
| `scripts/` | Start / stop the local vLLM server |

## Requirements

GeoEG runs in WSL (Ubuntu) with an NVIDIA GPU. It uses three conda environments:

- `geoeg`: the main pipeline (PyTorch with CUDA, transformers, GeoCLIP, SAM 2, Grounding DINO). Install with `pip install -r geoeg/requirements.txt`; the file header explains how to keep the core packages pinned.
- `geoeg-ocrtest`: PaddleOCR, called from the main pipeline as a persistent subprocess.
- `geoeg-vllm`: a vLLM server for Qwen2.5-VL-7B-Instruct-AWQ, started with `scripts/start_vllm.sh` (127.0.0.1:8001, model name `qwen2.5-vl-7b-awq`).

Model weights and datasets live outside the repository (Hugging Face cache, `~/geoeg_refs/` for the G3 code and checkpoint, `~/geoeg_data/` for evaluation data).

## Usage

```bash
conda activate geoeg
cd geoeg
bash ../scripts/start_vllm.sh                 # if vLLM is not already running on 8001
python -m geoeg.cli photo.jpg --mode local    # --json for the full result, --lang zh for Chinese output
python app.py                                 # Gradio UI at http://127.0.0.1:7861
python -m pytest                              # unit tests + end-to-end smoke test (skipped without vLLM/GPU)
```

`--mode hybrid` uses a cloud OpenAI-compatible API instead of the local VLM: copy `geoeg/.env.example` to `geoeg/.env` and fill in `GEOEG_API_BASE`, `GEOEG_API_MODEL` and `GEOEG_API_KEY`.

### Web demo

```bash
bash web/start_web.sh                         # starts vLLM if needed, then the web backend
```

Open http://localhost:7860, upload a photo or click one of the example images, and choose an engine: **GeoEG** (the full pipeline, with clue boxes, masks and the evidence graph drawn on the page) or **Baseline** (a single direct call to the local VLM). `bash scripts/stop_vllm.sh` frees the VLM's GPU memory afterwards.

### Output of a run

Each run is saved under `geoeg/runs/<time>_<image>/`:

| File | Contents |
|---|---|
| `result.json` | Top-5, explanation, all clues (with `bbox`, `mask_area`, `region`), edges, timing, GPU memory, baseline and decision features |
| `graph.json` | Evidence graph in NetworkX node-link format |
| `log.jsonl` | Per-stage event log: raw VLM output, boxes, OCR, evidence, decision |
| `crops/<clue id>.jpg` | Crop of each clue's box, with the area outside the mask dimmed |
| `masks/<clue id>.png` | SAM 2 mask, same size as the image (255 inside, 0 outside) |

Clue id prefixes: `c*` VLM description, `o*` OCR text, `e*` deterministic evidence.

### Configuration

Settings are in `geoeg/config.yaml`: output language, decision parameters (tuned by `run_eval tune`), per-module switches for ablations, the persistent OCR worker, and a GPU memory budget. All external calls (VLM, Nominatim, Overpass, OCR) are cached by request hash in `.cache/geoeg_cache.sqlite`.

## Evaluation

IM2GPS3K, sampled with seed 2026: 100 images for tuning (dev) and 200 held-out images for reporting (test).

```bash
python -m geoeg.eval.run_eval split
python -m geoeg.eval.run_eval run --split dev|test ...   # see the usage line at the top of run_eval.py
python -m geoeg.eval.run_eval tune      # grid search on the dev split only
python -m geoeg.eval.run_eval report    # test split only
```

For every test image, the error is the great-circle (haversine) distance between the top-1 predicted coordinates and the true location where the photo was taken. The table reports:

- **Median error**: the median of these distances over the 200 images (lower is better). Half of the predictions are closer than this, half are farther.
- **1 km / 25 km / 200 km / 750 km / 2500 km** (Acc@X km): the percentage of images whose error is at most X km (higher is better). These are the standard geolocation thresholds, roughly street, city, region, country and continent level. For example, 31.5% at 25 km means 63 of the 200 images were placed within 25 km of the true location. An image without a prediction counts as a miss.

The methods compared are:

- **Qwen2.5-VL-7B direct prediction**: the VLM alone, asked once for coordinates (the same prompt as the web demo's Baseline engine).
- **GeoCLIP alone**: the top-1 location from GeoCLIP image-to-GPS retrieval, with no VLM or evidence.
- **GeoEG**: the full pipeline described above.

Results on the 200-image test split:

| Method | Median error | 1 km | 25 km | 200 km | 750 km | 2500 km |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B direct prediction | 263 km | 4.5% | 28.0% | 49.0% | 65.0% | 77.0% |
| GeoCLIP alone | 210 km | 12.0% | 30.0% | 49.5% | 64.0% | 78.5% |
| **GeoEG** | **204 km** | **13.0%** | **31.5%** | **50.0%** | 65.0% | 78.0% |

Per-image errors, the tuning grid and the tuned parameters are in `geoeg/geoeg/eval/results/`. With 200 images the differences are within roughly ±3 percentage points, and the 7B VLM is the main bottleneck; stronger VLMs (via hybrid mode) are the natural next step.

## Third-party components

| Component | Use | Licence |
|---|---|---|
| [geo-sleuth](https://github.com/Oldcircle/geo-sleuth) | Scripts and lookup data copied unmodified into `geoeg/geoeg/tools/geosleuth/`, called as a subprocess | MIT (see `LICENSE` there) |
| [G3](https://github.com/Applied-Machine-Learning-Lab/G3) | Imported from a local checkout, not copied; checkpoint from HF `Jia-py/G3-checkpoint` | Apache-2.0 |
| [GeoCLIP](https://github.com/VicenteVivan/geo-clip) | pip dependency | MIT |
| [SAM 2](https://github.com/facebookresearch/sam2) | Dependency; weights `facebook/sam2.1-hiera-small` | Apache-2.0 |
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) | transformers implementation and weights | Apache-2.0 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Separate environment, subprocess | Apache-2.0 |
| Qwen2.5-VL-7B-Instruct-AWQ | Served by vLLM | Qwen licence (see model card) |
| OpenStreetMap / Nominatim / Overpass | Online services, used with a custom User-Agent, ≤1 request/s and caching | ODbL data |

IM2GPS3K images are not distributed with this repository; their copyright belongs to the original authors and they are used for research evaluation only.
