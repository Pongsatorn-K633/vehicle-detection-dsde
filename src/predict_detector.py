"""Run a trained detector (YOLO26 .pt or RF-DETR .pth) and save its raw predictions.

    python src/predict_detector.py --weights runs/y26m_640/weights/best.pt            # val + test
    python src/predict_detector.py --weights runs/rfdm_640/checkpoint_best_total.pth --splits test

For each split it writes two files in preds/<name>/ (name = run folder unless --name is given):
  <split>.csv       predictions on the original images
  <split>_flip.csv  predictions on the mirrored images, mapped back to original coordinates (for TTA)
'val' = labelled images of the validation cameras, 'test' = the Kaggle test images.
A low score threshold is used on purpose: low-ranked boxes cannot lower AP, and fusion needs them.
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import CLASS_NAMES, PRED_COLUMNS, ROOT, split_images, write_preds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--name", default=None, help="output folder under preds/ (default: run folder name)")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640, help="YOLO only; RF-DETR uses its training resolution")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--no-flip", action="store_true", help="skip the mirrored-image pass")
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


class YoloDetector:
    def __init__(self, weights, imgsz, conf, max_det):
        from ultralytics import YOLO
        self.model, self.kw = YOLO(str(weights)), dict(imgsz=imgsz, conf=conf, max_det=max_det, verbose=False)

    def __call__(self, images):
        """images: list of RGB uint8 arrays -> list of (xyxy, scores, classes)."""
        results = self.model.predict([im[:, :, ::-1] for im in images], **self.kw)  # Ultralytics wants BGR
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
                for r in results]


class RFDETRDetector:
    def __init__(self, weights, conf, max_det):
        from rfdetr import RFDETR
        self.model, self.conf, self.max_det = RFDETR.from_checkpoint(str(weights)), conf, max_det

    def __call__(self, images):
        dets = self.model.predict([Image.fromarray(im) for im in images], threshold=self.conf,
                                  include_source_image=False)
        dets = dets if isinstance(dets, list) else [dets]
        out = []
        for d in dets:
            # RF-DETR's head has one slot more than the 8 classes; drop anything outside 0..7
            valid = np.flatnonzero(d.class_id < len(CLASS_NAMES))
            keep = valid[np.argsort(-d.confidence[valid])][:self.max_det]
            out.append((d.xyxy[keep], d.confidence[keep], d.class_id[keep].astype(int)))
        return out


def run(detector, img_dir, names, flip, batch):
    rows = []
    for i in tqdm(range(0, len(names), batch), desc="flip" if flip else "orig", leave=False):
        chunk = names[i:i + batch]
        images = [np.asarray(Image.open(img_dir / f).convert("RGB")) for f in chunk]
        if flip:
            images = [np.ascontiguousarray(im[:, ::-1]) for im in images]
        for f, im, (xyxy, scores, classes) in zip(chunk, images, detector(images)):
            xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
            if flip:  # mirror the boxes back: x1' = W - x2, x2' = W - x1
                w = im.shape[1]
                xyxy[:, [0, 2]] = w - xyxy[:, [2, 0]]
            for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes):
                rows.append((f, int(c), float(s), x1, y1, x2, y2))
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def main():
    args = parse_args()
    weights = ROOT / args.weights
    if weights.suffix == ".pt":
        detector = YoloDetector(weights, args.imgsz, args.conf, args.max_det)
        name = args.name or weights.parent.parent.name  # runs/<name>/weights/best.pt
    elif weights.suffix == ".pth":
        detector = RFDETRDetector(weights, args.conf, args.max_det)
        name = args.name or weights.parent.name  # runs/<name>/checkpoint_best_total.pth
    else:
        raise ValueError("--weights must be a YOLO .pt or an RF-DETR .pth file")

    for split in args.splits:
        img_dir, names = split_images(split)
        for flip in [False] if args.no_flip else [False, True]:
            df = run(detector, img_dir, names, flip, args.batch)
            out = write_preds(df, name, split, flip)
            print(f"{split}{' (flipped)' if flip else ''}: {len(df)} boxes on {len(names)} images -> {out}")


if __name__ == "__main__":
    main()
