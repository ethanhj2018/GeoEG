"""Standalone PaddleOCR worker. Runs inside the geoeg-ocrtest conda env (paddle is NOT installed in geoeg).

One-shot:  python ocr_worker.py --in request.json --out result.json
Serve:     python -u ocr_worker.py --serve [--idle-exit 600]
           reads one JSON request per stdin line, answers one line "@@GEOEG_OCR@@<json>" on stdout (Paddle's own
           log lines are ignored by the client), keeps the model warm, frees cached GPU memory after each request
           and exits after `--idle-exit` seconds without requests.
request: {"images": [path, ...], "det_model": "...", "rec_model": "..."}
result:  {"results": [{"image": path, "lines": [{"text": str, "score": float, "box": [x0,y0,x1,y1]}]}]}
Must not import anything from the geoeg package.
"""
import argparse
import json
import select
import sys
import time

MARK = "@@GEOEG_OCR@@"


def make_ocr(req: dict):
    from paddleocr import PaddleOCR
    return PaddleOCR(text_detection_model_name=req["det_model"], text_recognition_model_name=req["rec_model"],
                     use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)


def recognise(ocr, images: list[str]) -> list[dict]:
    results = []
    for path in images:
        r = ocr.predict(path)[0]
        lines = [{"text": text, "score": round(float(score), 3), "box": [float(v) for v in box]}
                 for text, score, box in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"])]
        results.append({"image": path, "lines": lines})
    return results


def serve(idle_exit: float) -> None:
    ocr, models = None, None
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], idle_exit)
        if not ready:
            return                                    # idle: exit and release the GPU
        line = sys.stdin.readline()
        if not line:
            return                                    # client closed the pipe
        t0 = time.time()
        try:
            req = json.loads(line)
            if ocr is None or models != (req["det_model"], req["rec_model"]):
                ocr, models = make_ocr(req), (req["det_model"], req["rec_model"])
            resp = {"results": recognise(ocr, req["images"]), "elapsed_s": round(time.time() - t0, 3)}
        except Exception as e:                        # report, keep serving
            resp = {"error": repr(e)}
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
        sys.stdout.write(MARK + json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--idle-exit", type=float, default=600)
    args = ap.parse_args()
    if args.serve:
        serve(args.idle_exit)
        return
    req = json.load(open(args.inp, encoding="utf-8"))
    t0 = time.time()
    results = recognise(make_ocr(req), req["images"])
    json.dump({"results": results, "elapsed_s": round(time.time() - t0, 2)},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
