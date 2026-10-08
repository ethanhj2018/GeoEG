"""Standalone PaddleOCR worker. Runs inside the geoeg-ocrtest conda env (paddle is NOT installed in geoeg).

Usage: python ocr_worker.py --in request.json --out result.json
request: {"images": [path, ...], "det_model": "...", "rec_model": "..."}
result:  {"results": [{"image": path, "lines": [{"text": str, "score": float, "box": [x0,y0,x1,y1]}]}]}
Must not import anything from the geoeg package.
"""
import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    req = json.load(open(args.inp, encoding="utf-8"))
    t0 = time.time()
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(text_detection_model_name=req["det_model"], text_recognition_model_name=req["rec_model"],
                    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    results = []
    for path in req["images"]:
        r = ocr.predict(path)[0]
        lines = []
        for text, score, box in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"]):
            lines.append({"text": text, "score": round(float(score), 3), "box": [float(v) for v in box]})
        results.append({"image": path, "lines": lines})
    json.dump({"results": results, "elapsed_s": round(time.time() - t0, 2)},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
