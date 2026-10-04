"""Merge predictions of several detectors (and their flipped TTA pass) with Weighted Boxes Fusion.

    python src/fuse.py --runs rfdm_640 y26m_640 --weights 2 1 --out wbf_exp                 # val + test
    python src/fuse.py --runs rfdm_640_full y26m_640_full --weights 2 1 --out wbf_final --splits test
    python src/fuse.py --runs y26m_640 --out y26m_640_tta                                  # TTA only

Every run contributes its <split>.csv and, unless --no-tta, its <split>_flip.csv, both with that run's
weight. Each source first gets per-class NMS (--pre-nms): detectors emit many weak near-duplicates of the
same object, and WBF would average those into the good box and drag its score and position down.
Boxes are normalised to 0..1 for WBF and mapped back to pixels. Output: preds/<out>/<split>.csv.
"""
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from torchvision.ops import batched_nms
from tqdm import tqdm

from common import PRED_COLUMNS, image_dir_of, read_preds, write_preds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="prediction folders under preds/")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="one per run (default all 1)")
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--no-tta", action="store_true", help="ignore the *_flip.csv files")
    p.add_argument("--iou-thr", type=float, default=0.6)
    p.add_argument("--skip-thr", type=float, default=0.01, help="drop input boxes below this score")
    p.add_argument("--pre-nms", type=float, default=0.6, help="per-class NMS IoU inside each source (0 = off)")
    p.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg", "absent_model_aware_avg"])
    return p.parse_args()


def nms_per_class(g, iou):
    if iou <= 0 or len(g) < 2:
        return g
    keep = batched_nms(torch.as_tensor(g[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float32")),
                       torch.as_tensor(g.score.to_numpy(dtype="float32")),
                       torch.as_tensor(g.class_id.to_numpy()), iou)
    return g.iloc[keep.numpy()]


@lru_cache(maxsize=None)
def image_size(fname):
    with Image.open(image_dir_of(fname) / fname) as im:
        return im.size  # (W, H)


def fuse_split(args, split, weights):
    sources, source_w = [], []
    for run, w in zip(args.runs, weights):
        for flip in [False] if args.no_tta else [False, True]:
            sources.append(read_preds(run, split, flip))
            source_w.append(w)
    images = sorted(set().union(*(set(s.image) for s in sources)))
    by_image = [{k: g for k, g in s[s.score >= args.skip_thr].groupby("image")} for s in sources]

    rows = []
    for f in tqdm(images, desc=split, leave=False):
        W, H = image_size(f)
        scale = np.array([W, H, W, H], dtype=float)
        boxes, scores, labels = [], [], []
        for src in by_image:
            g = nms_per_class(src.get(f, sources[0].iloc[:0]), args.pre_nms)  # no box here -> empty list
            boxes.append(np.clip(g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) / scale, 0, 1))
            scores.append(g.score.to_numpy())
            labels.append(g.class_id.to_numpy())
        b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=source_w, iou_thr=args.iou_thr,
                                        skip_box_thr=args.skip_thr, conf_type=args.conf_type)
        b = b * scale
        rows += [(f, int(c), float(sc), *bb) for bb, sc, c in zip(b, s, l)]
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def main():
    args = parse_args()
    weights = args.weights or [1.0] * len(args.runs)
    if len(weights) != len(args.runs):
        raise ValueError("give one --weights value per --runs entry")
    for split in args.splits:
        df = fuse_split(args, split, weights)
        out = write_preds(df, args.out, split)
        print(f"{split}: {len(df)} fused boxes -> {out}")


if __name__ == "__main__":
    main()
