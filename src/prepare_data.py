"""Convert train.csv into training datasets with a camera-held-out validation split.

    python src/prepare_data.py           # train = 12 cameras, val = VAL_CAMERAS
    python src/prepare_data.py --full    # train = all 15 cameras (for the final models)

Writes two copies of the same split:
  datasets/yolo[_full]/    Ultralytics format: images/{train,val}, labels/{train,val}, data.yaml
  datasets/rfdetr[_full]/  RF-DETR (Roboflow COCO) format: {train,valid}/ + _annotations.coco.json
Images without any box are kept as background images.

Rare classes are oversampled with repeat-factor sampling (LVIS, Gupta et al. 2019): class c gets
r_c = max(1, sqrt(t / f_c)), f_c = fraction of train images that contain c, and each train image is copied
round(max r_c of its classes) times (at most 4), as '<name>__rep<k>.jpg'. All boxes of a copied image stay
labelled, so no vehicle is turned into background. Validation images are never copied. --repeat-thr 0 = off.
"""
import argparse
import json
import math
import shutil

import pandas as pd
import yaml

from common import CLASS_NAMES, DATASETS_DIR, TRAIN_IMG_DIR, VAL_CAMERAS, camera_of, load_gt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cameras", nargs="+", default=VAL_CAMERAS)
    p.add_argument("--full", action="store_true",
                   help="put every camera in train; val cameras are still used for monitoring only")
    p.add_argument("--repeat-thr", type=float, default=0.3, help="t in repeat-factor sampling (0 = no oversampling)")
    p.add_argument("--max-repeat", type=int, default=4)
    return p.parse_args()


def repeat_counts(names, df, thr, max_repeat):
    """Copies of each train image under repeat-factor sampling."""
    if thr <= 0:
        return {f: 1 for f in names}
    d = df[df.image_id.isin(set(names))]
    frac = d.groupby("class_id").image_id.nunique() / len(names)
    r_cls = {c: max(1.0, math.sqrt(thr / f)) for c, f in frac.items()}
    r_img = d.groupby("image_id").class_id.agg(lambda cs: max(r_cls[c] for c in set(cs)))
    return {f: int(min(max_repeat, round(r_img.get(f, 1.0)))) for f in names}


def write_yolo(out, splits, boxes_by_img):
    for split, names in splits.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for src, f in names:
            shutil.copy2(TRAIN_IMG_DIR / src, img_dir / f)
            g = boxes_by_img.get(src)
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
        for img_id, (src, f) in enumerate(names):
            shutil.copy2(TRAIN_IMG_DIR / src, d / f)
            coco["images"].append({"id": img_id, "file_name": f, "width": 352, "height": 288})
            g = boxes_by_img.get(src)
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

    val = [f for f in files if camera_of(f) in val_cams]
    train = [f for f in files if args.full or camera_of(f) not in val_cams]
    reps = repeat_counts(train, df, args.repeat_thr, args.max_repeat)
    # (source image, written file name); copies get a '__rep<k>' suffix
    splits = {"train": [(f, f if k == 0 else f.replace(".jpg", f"__rep{k}.jpg")) for f in train for k in range(reps[f])],
              "val": [(f, f) for f in val]}

    for name, writer in [("yolo", write_yolo), ("rfdetr", write_rfdetr)]:
        out = DATASETS_DIR / f"{name}{suffix}"
        if out.exists():
            shutil.rmtree(out)
        writer(out, splits, boxes_by_img)
        print(f"Wrote {out}")

    stats = {}
    for split, names in splits.items():
        copies = pd.Series([src for src, _ in names]).value_counts()
        counts = df.assign(n=df.image_id.map(copies)).dropna(subset=["n"]).groupby("class_id").n.sum()
        stats[split] = [len(names)] + [int(counts.get(i, 0)) for i in range(len(CLASS_NAMES))]
    print(f"Val cameras: {sorted(val_cams)}{'  (full mode: val cameras are also in train)' if args.full else ''}")
    print(f"Oversampling: t={args.repeat_thr}, {len(train)} train images -> {len(splits['train'])} after copies")
    print(pd.DataFrame(stats, index=["images"] + CLASS_NAMES).to_string())


if __name__ == "__main__":
    main()
