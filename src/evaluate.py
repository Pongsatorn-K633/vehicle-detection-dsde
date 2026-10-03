"""Score a model on the validation cameras with pycocotools mAP@50 (same metric as Kaggle).

    python src/evaluate.py --weights runs/y26m_640/weights/best.pt

Prints per-class AP50, the 8-class mAP@50, the class-agnostic AP50 (localization only)
and a confidence-threshold sweep. Predictions and results are saved to runs/<name>/eval/.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from common import CLASS_NAMES, ROOT, YOLO_DIR, predict_images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data-dir", default=str(YOLO_DIR), help="dataset made by prepare_data.py")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--confs", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.25])
    p.add_argument("--augment", action="store_true", help="test-time augmentation")
    p.add_argument("--device", default="0")
    return p.parse_args()


def coco_ap50(gt, dets, use_cats=True):
    """Return (mAP50, per-class AP50 list). Classes without ground truth are NaN and skipped."""
    k = len(CLASS_NAMES) if use_cats else 1
    if not dets:
        return 0.0, [0.0] * k
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.useCats = int(use_cats)
        ev.evaluate()
        ev.accumulate()
    # precision[T, R, K, A, M]: IoU=0.50, all recall points, class k, area=all, maxDets=100
    prec = ev.eval["precision"][0, :, :, 0, -1]
    per_class = [float(prec[:, c][prec[:, c] > -1].mean()) if (prec[:, c] > -1).any() else float("nan")
                 for c in range(prec.shape[1])]
    return float(np.nanmean(per_class)), per_class


def main():
    args = parse_args()
    weights = ROOT / args.weights
    data_dir = ROOT / args.data_dir
    out_dir = weights.parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(data_dir / "val_coco.json"))
    id_of = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    paths = [data_dir / "images" / "val" / f for f in id_of]

    model = YOLO(str(weights))
    dets = []
    for fname, xyxy, scores, classes in predict_images(
            model, paths, args.imgsz, min(args.confs), args.max_det, args.augment, device=args.device):
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes):
            dets.append({"image_id": id_of[fname], "category_id": int(c),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    (out_dir / "val_preds.json").write_text(json.dumps(dets))

    rows = []
    for conf in sorted(args.confs):
        kept = [d for d in dets if d["score"] >= conf]
        m, per_class = coco_ap50(gt, kept)
        agnostic, _ = coco_ap50(gt, kept, use_cats=False)
        rows.append({"conf": conf, "mAP50": m, "agnostic_AP50": agnostic,
                     **dict(zip(CLASS_NAMES, per_class))})
    table = pd.DataFrame(rows).set_index("conf")
    table.to_csv(out_dir / "ap50.csv")

    n_gt = pd.Series([a["category_id"] for a in gt.dataset["annotations"]]).value_counts()
    print(f"Val images: {len(paths)}  |  GT boxes per class: "
          + ", ".join(f"{n}={n_gt.get(i, 0)}" for i, n in enumerate(CLASS_NAMES)))
    print(table.T.round(4).to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
