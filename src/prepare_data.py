"""Convert train.csv to an Ultralytics YOLO dataset with a camera-held-out validation split.

    python src/prepare_data.py                 # train = 12 cameras, val = VAL_CAMERAS
    python src/prepare_data.py --full          # train = all 15 cameras (for the final model)

Writes <out>/images/{train,val}, <out>/labels/{train,val}, <out>/data.yaml and
<out>/val_coco.json (ground truth for src/evaluate.py). Images without any box are
kept as background images (empty label file).
"""
import argparse
import json
import shutil

import pandas as pd
import yaml
from PIL import Image

from common import CLASS_NAMES, ROOT, TRAIN_CSV, TRAIN_IMG_DIR, VAL_CAMERAS, YOLO_DIR, camera_of


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cameras", nargs="+", default=VAL_CAMERAS)
    p.add_argument("--full", action="store_true",
                   help="put every camera in train; val cameras are still used for monitoring only")
    p.add_argument("--out", default=None, help="output dir (default datasets/yolo or datasets/yolo_full)")
    return p.parse_args()


def main():
    args = parse_args()
    out = ROOT / args.out if args.out else (YOLO_DIR.with_name("yolo_full") if args.full else YOLO_DIR)
    if out.exists():
        shutil.rmtree(out)

    df = pd.read_csv(TRAIN_CSV)
    boxes_by_img = {k: g for k, g in df.groupby("image_id")}
    files = sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpg"))
    val_cams = set(args.val_cameras)

    splits = {"train": [], "val": []}
    for f in files:
        in_val = camera_of(f) in val_cams
        if in_val:
            splits["val"].append(f)
        if args.full or not in_val:
            splits["train"].append(f)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]}
    stats = {}
    for split, names in splits.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        counts = [0] * len(CLASS_NAMES)
        for f in names:
            shutil.copy2(TRAIN_IMG_DIR / f, img_dir / f)
            w, h = Image.open(TRAIN_IMG_DIR / f).size
            g = boxes_by_img.get(f)
            lines = []
            if split == "val":
                img_id = len(coco["images"])
                coco["images"].append({"id": img_id, "file_name": f, "width": w, "height": h})
            if g is not None:
                for r in g.itertuples():
                    x1, y1 = max(0, r.x1), max(0, r.y1)
                    x2, y2 = min(w, r.x2), min(h, r.y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    counts[r.class_id] += 1
                    lines.append(f"{r.class_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                 f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    if split == "val":
                        coco["annotations"].append({
                            "id": len(coco["annotations"]), "image_id": img_id, "category_id": int(r.class_id),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
                        })
            (lbl_dir / f).with_suffix(".txt").write_text("\n".join(lines))
        stats[split] = [len(names)] + counts

    (out / "val_coco.json").write_text(json.dumps(coco))
    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                 "names": dict(enumerate(CLASS_NAMES))}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    table = pd.DataFrame(stats, index=["images"] + CLASS_NAMES)
    print(f"Wrote {out}  (val cameras: {sorted(val_cams)}{', full mode' if args.full else ''})")
    print(table.to_string())


if __name__ == "__main__":
    main()
