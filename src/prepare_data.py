"""Convert train.csv into training datasets with a camera-held-out validation split.

    python src/prepare_data.py           # train = 12 cameras, val = VAL_CAMERAS
    python src/prepare_data.py --full    # train = all 15 cameras (for the final models)

Writes two copies of the same split:
  datasets/yolo[_full]/    Ultralytics format: images/{train,val}, labels/{train,val}, data.yaml
  datasets/rfdetr[_full]/  RF-DETR (Roboflow COCO) format: {train,valid}/ + _annotations.coco.json
Images without any box are kept as background images.
"""
import argparse
import json
import shutil

import pandas as pd
import yaml

from common import CLASS_NAMES, DATASETS_DIR, TRAIN_IMG_DIR, VAL_CAMERAS, camera_of, load_gt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cameras", nargs="+", default=VAL_CAMERAS)
    p.add_argument("--full", action="store_true",
                   help="put every camera in train; val cameras are still used for monitoring only")
    return p.parse_args()


def write_yolo(out, splits, boxes_by_img):
    for split, names in splits.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for f in names:
            shutil.copy2(TRAIN_IMG_DIR / f, img_dir / f)
            g = boxes_by_img.get(f)
            lines = [] if g is None else [
                f"{r.class_id} {(r.x1 + r.x2) / 2 / 352:.6f} {(r.y1 + r.y2) / 2 / 288:.6f} "
                f"{(r.x2 - r.x1) / 352:.6f} {(r.y2 - r.y1) / 288:.6f}" for r in g.itertuples()]
            (lbl_dir / f).with_suffix(".txt").write_text("\n".join(lines))
    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                 "names": dict(enumerate(CLASS_NAMES))}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))


def write_rfdetr(out, splits, boxes_by_img):
    # Category ids 0..7 with a placeholder supercategory: RF-DETR then keeps class_id == category_id
    # (no Roboflow-style parent category that would shift every id by one).
    categories = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(CLASS_NAMES)]
    for split, names in splits.items():
        d = out / ("valid" if split == "val" else split)
        d.mkdir(parents=True)
        coco = {"images": [], "annotations": [], "categories": categories}
        for img_id, f in enumerate(names):
            shutil.copy2(TRAIN_IMG_DIR / f, d / f)
            coco["images"].append({"id": img_id, "file_name": f, "width": 352, "height": 288})
            g = boxes_by_img.get(f)
            for r in ([] if g is None else g.itertuples()):
                w, h = float(r.x2 - r.x1), float(r.y2 - r.y1)
                coco["annotations"].append({
                    "id": len(coco["annotations"]), "image_id": img_id, "category_id": int(r.class_id),
                    "bbox": [float(r.x1), float(r.y1), w, h], "area": w * h, "iscrowd": 0})
        (d / "_annotations.coco.json").write_text(json.dumps(coco))


def main():
    args = parse_args()
    suffix = "_full" if args.full else ""
    df = load_gt()
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

    for name, writer in [("yolo", write_yolo), ("rfdetr", write_rfdetr)]:
        out = DATASETS_DIR / f"{name}{suffix}"
        if out.exists():
            shutil.rmtree(out)
        writer(out, splits, boxes_by_img)
        print(f"Wrote {out}")

    stats = {}
    for split, names in splits.items():
        counts = df[df.image_id.isin(set(names))].class_id.value_counts()
        stats[split] = [len(names)] + [int(counts.get(i, 0)) for i in range(len(CLASS_NAMES))]
    print(f"Val cameras: {sorted(val_cams)}{'  (full mode: val cameras are also in train)' if args.full else ''}")
    print(pd.DataFrame(stats, index=["images"] + CLASS_NAMES).to_string())


if __name__ == "__main__":
    main()
