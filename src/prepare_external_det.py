"""Turn the CCTV-like external images into detector training frames at our scale, with missing vehicles filled in.

    python src/prepare_external_det.py --rfdetr runs/rfdl_e12/checkpoint_best_total.pth \
        --yolo runs/y26m_e25/weights/last.pt --out datasets/external_det          # for the round 1 experiment
    python src/prepare_external_det.py --rfdetr runs/rfdl_e12_full/last_ema.pth \
        --yolo runs/y26m_e25_full/weights/last.pt --out datasets/external_det_full # for the final models

Input: datasets/external/external.csv (prepare_external.py), only the CCTV-like sets (vehicle_car, traffic_count).
- Close-ups: images whose largest box is above --max-rel of the image side (product photos in vehicle_car) are
  skipped.
- Scale: their vehicles are 2-3x larger relative to the image than ours (median box 0.10-0.14 of the image side
  vs 0.05). Each image is resized to 176x144 and 4 images of the same set are tiled into one 352x288 frame, the
  size of our images, which halves the relative size and gives about 10 vehicles per frame, like ours.
- Missing labels: these sets label the near vehicles and skip the small far ones, which would teach the detector
  that far vehicles are background. Our detectors (RF-DETR + YOLO, fused 2:1) run on each frame; a predicted box
  with score >= --pseudo-thr (0.35) that overlaps no labelled box (IoU < --free-iou) is added (pseudo=1).
  Use detectors that never saw the validation cameras when the frames feed a validation experiment.
Writes <out>/images/*.jpg and <out>/boxes.csv (image_id, class_id, x1, y1, x2, y2, dataset, pseudo).
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from torchvision.ops import box_iou

from common import DATASETS_DIR, ROOT
from predict_detector import RFDETRDetector, YoloDetector

W, H = 352, 288
TW, TH = W // 2, H // 2
CCTV_SETS = ["vehicle_car", "traffic_count"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rfdetr", required=True)
    p.add_argument("--yolo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pseudo-thr", type=float, default=0.35)
    p.add_argument("--max-rel", type=float, default=0.35,
                   help="skip source images whose largest box is above this fraction of the image side (close-ups)")
    p.add_argument("--free-iou", type=float, default=0.3)
    p.add_argument("--min-side", type=float, default=3.0, help="drop boxes smaller than this (pixels) after resizing")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_tiles(ext, out_img, seed, max_rel):
    """Tile 4 images of the same set into one 352x288 frame; returns the labelled boxes in frame pixels."""
    rows = []
    rng = random.Random(seed)
    sizes = {f: Image.open(DATASETS_DIR / "external" / "images" / f).size for f in ext.image_id.unique()}
    rel = np.sqrt((ext.x2 - ext.x1) * (ext.y2 - ext.y1) / ext.image_id.map(lambda f: sizes[f][0] * sizes[f][1]))
    close_up = set(ext[rel > max_rel].image_id)
    print(f"close-up images skipped: {len(close_up)} of {len(sizes)}")
    for ds in CCTV_SETS:
        names = sorted(set(ext[ext.dataset == ds].image_id) - close_up)
        rng.shuffle(names)
        by_img = {k: g for k, g in ext[ext.dataset == ds].groupby("image_id")}
        for t in range(0, len(names) - 3, 4):
            frame = Image.new("RGB", (W, H))
            fname = f"ext_{ds}_{t // 4:04d}.jpg"
            for k, src in enumerate(names[t:t + 4]):
                im = Image.open(DATASETS_DIR / "external" / "images" / src).convert("RGB")
                sx, sy = TW / im.width, TH / im.height
                ox, oy = (k % 2) * TW, (k // 2) * TH
                frame.paste(im.resize((TW, TH), Image.BILINEAR), (ox, oy))
                for r in by_img[src].itertuples():
                    rows.append(dict(image_id=fname, class_id=int(r.class_id), x1=ox + r.x1 * sx, y1=oy + r.y1 * sy,
                                     x2=ox + r.x2 * sx, y2=oy + r.y2 * sy, dataset=ds, pseudo=0))
            frame.save(out_img / fname, quality=95)
    return pd.DataFrame(rows)


def pseudo_boxes(frames, img_dir, rfdetr, yolo, thr):
    """Fused (RF-DETR x2, YOLO x1) predictions with score >= thr on each frame."""
    out = []
    scale = np.array([W, H, W, H], dtype=float)
    for i in range(0, len(frames), 16):
        chunk = frames[i:i + 16]
        images = [np.asarray(Image.open(img_dir / f).convert("RGB")) for f in chunk]
        for f, a, b in zip(chunk, rfdetr(images), yolo(images)):
            boxes = [np.clip(np.asarray(d[0], dtype=float).reshape(-1, 4) / scale, 0, 1) for d in (a, b)]
            fb, fs, fl = weighted_boxes_fusion(boxes, [a[1], b[1]], [a[2], b[2]], weights=[2, 1], iou_thr=0.6,
                                               skip_box_thr=0.05)
            for bb, s, c in zip(fb * scale, fs, fl):
                if s >= thr:
                    out.append(dict(image_id=f, class_id=int(c), x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3], score=s))
    return pd.DataFrame(out)


def main():
    args = parse_args()
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)

    ext = pd.read_csv(DATASETS_DIR / "external" / "external.csv")
    ext = ext[ext.dataset.isin(CCTV_SETS)]
    gt = build_tiles(ext, out / "images", args.seed, args.max_rel)
    frames = sorted(gt.image_id.unique())

    rfdetr = RFDETRDetector(ROOT / args.rfdetr, conf=0.05, max_det=300)
    yolo = YoloDetector(ROOT / args.yolo, imgsz=704, conf=0.05, max_det=300)
    pred = pseudo_boxes(frames, out / "images", rfdetr, yolo, args.pseudo_thr)

    added = []
    for f, p in pred.groupby("image_id"):
        g = gt[gt.image_id == f]
        pb = torch.as_tensor(p[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float32"))
        if len(g):
            gb = torch.as_tensor(g[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float32"))
            free = box_iou(pb, gb).max(1).values < args.free_iou
        else:
            free = torch.ones(len(p), dtype=torch.bool)
        added.append(p[free.numpy()].drop(columns="score").assign(dataset=g.dataset.iloc[0] if len(g) else "", pseudo=1))
    boxes = pd.concat([gt] + added, ignore_index=True)
    boxes = boxes[((boxes.x2 - boxes.x1) >= args.min_side) & ((boxes.y2 - boxes.y1) >= args.min_side)]
    for col in ["x1", "y1", "x2", "y2"]:
        boxes[col] = boxes[col].clip(0, W if col[0] == "x" else H).round(1)
    boxes.to_csv(out / "boxes.csv", index=False)

    print(f"{len(frames)} frames, {len(boxes)} boxes ({int(boxes.pseudo.sum())} filled in) -> {out}")
    print(f"boxes per frame: {len(boxes) / len(frames):.1f}; relative size median "
          f"{np.sqrt((boxes.x2 - boxes.x1) * (boxes.y2 - boxes.y1) / (W * H)).median():.3f} (ours 0.050)")
    print(boxes.pivot_table(index="pseudo", columns="class_id", values="x1", aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()
