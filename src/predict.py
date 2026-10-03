"""Predict the test set and write a Kaggle submission CSV.

    python src/predict.py --weights runs/y26m_640_full/weights/best.pt --out submissions/y26m_640_full.csv

Test files are named '<cam>_<date>_<time>.jpg' but the submission needs the long
image_id from sample_submission.csv, so ids are matched on camera + timestamp.
Test images that are not in sample_submission.csv (16 of them) are skipped.
"""
import argparse

import pandas as pd
from ultralytics import YOLO

from common import ROOT, SAMPLE_SUB_CSV, SUBMISSION_DIR, TEST_IMG_DIR, predict_images, short_test_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default=None, help="default submissions/<run name>.csv")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--augment", action="store_true", help="test-time augmentation")
    p.add_argument("--device", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    weights = ROOT / args.weights
    out = ROOT / args.out if args.out else SUBMISSION_DIR / f"{weights.parent.parent.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE_SUB_CSV, encoding="utf-8")
    long_id = {short_test_name(i): i for i in sample.image_id.unique()}
    missing = [f for f in long_id if not (TEST_IMG_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} submission images not found in {TEST_IMG_DIR}, e.g. {missing[:3]}")

    model = YOLO(str(weights))
    rows = []
    for fname, xyxy, scores, classes in predict_images(
            model, [TEST_IMG_DIR / f for f in long_id], args.imgsz, args.conf, args.max_det,
            args.augment, device=args.device):
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes):
            rows.append((long_id[fname], int(c), round(float(s), 5),
                         round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)))

    sub = pd.DataFrame(rows, columns=["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    sub.insert(0, "id", range(len(sub)))
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(sub)} boxes for {sub.image_id.nunique()}/{len(long_id)} images -> {out}")
    print(sub.class_id.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
