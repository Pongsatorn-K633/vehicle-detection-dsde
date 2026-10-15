"""Convert the Roboflow datasets in external-data/ into one box table in our 8 classes.

    python src/prepare_external.py

Reads external-data/<set>/{train,valid,test}/{images,labels} (YOLO export), maps each set's class names to ours
with MAPPING below (decisions and reasons: EXTERNAL_DATA.md), and writes
  datasets/external/images/<set>__<name>.jpg   one copy per source image, long side at most --max-side
  datasets/external/external.csv               image_id, class_id, x1, y1, x2, y2 (pixels), dataset, cctv

- Roboflow's augmented copies ('<name>.rf.<hash>') are kept once; images repeated across sets are kept once
  (perceptual hash, CCTV-like sets first). Polygons become their bounding box.
- Class names not in MAPPING are dropped, including traffic_count's mixed labels (Car0 can be a car, pickup or
  truck). Images left without any box are skipped.
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from common import CLASS_ID, DATASETS_DIR, ROOT

EXT_DIR = ROOT / "external-data"
OUT_DIR = DATASETS_DIR / "external"
RF = re.compile(r"\.rf\.[0-9a-f]+$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

C, M, B, T, K, V, P, S = "Car", "Motorcycle", "Bus", "Truck", "Tuktuk", "Van", "Pickup", "Songthaew"
# Every name not listed is dropped. CCTV-like sets come first so cross-set duplicates keep their copy.
MAPPING = {
    "vehicle_car": {
        "sedan": C, "SUV": C, "PPV": C, "Motorcycle": M, "bus": B, "truck": T, "Solid box pick-up": T,
        "van": V, "MPV": V, "pick-up": P, "Songthaew": S},
    "traffic_count": {
        "Car1": C, "Motorcycle": M, "Bus_L": B, "Truck_L": T, "Truck_m": T, "Trailer": T, "Van": V, "Songthaew": S},
    "vehicle_detection_v22": {
        "sedan": C, "hatchback": C, "suv": C, "jeep": C, "taxi": C, "motorcycle": M, "bus": B, "truck": T,
        "tuktuk": K, "van": V, "pickup": P, "songthaew": S},
    "vehicle_detection_v1": {
        "sedan": C, "hatchback": C, "suv": C, "jeep": C, "taxi": C, "supercar": C, "motorcycle": M, "bus": B,
        "truck": T, "tuktuk": K, "van": V, "pickup": P, "songthaew": S},
    "classification_of_cars": {
        "car": C, "SUV": C, "motorbike": M, "bus": B, "10 wheel large truck": T, "6 wheel medium truck": T,
        "semi trailer truck": T, "tow truck": T, "Six-wheeled vehicle bus": T, "Two-section pickup truck": T,
        "truck": P, "van": V, "songthaew": S},
    "songthaew": {"songthaew": S},
    "test_mlejl": {"Car": C, "SUV": C, "MC": M, "3WTuktuk": K, "Songthaew": S},
    "tuktuk_detection": {"tuktuk": K},
}
CCTV_SETS = {"vehicle_car", "traffic_count"}
# traffic_count's mixed labels (Car0 can be a car, pickup or truck) give no box, but a frame that has them still
# claims its picture, so a near-identical later frame is skipped as a duplicate
MIXED = {"Car0", "Truck_s", "Bus_s"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=list(MAPPING), choices=list(MAPPING),
                   help="only these datasets (default: all, in MAPPING order)")
    p.add_argument("--max-side", type=int, default=1280)
    return p.parse_args()


def ahash(im):
    a = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32)
    return np.packbits(a > a.mean()).tobytes()


def read_boxes(lab, W, H):
    """YOLO label file -> [(class index, x1, y1, x2, y2 in pixels)]; polygons become their bounding box."""
    out = []
    for line in open(lab) if lab.exists() else []:
        v = line.split()
        if not v:
            continue
        vals = list(map(float, v[1:]))
        if len(vals) > 4:
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = vals[0] - vals[2] / 2, vals[1] - vals[3] / 2, vals[0] + vals[2] / 2, vals[1] + vals[3] / 2
        x1, y1, x2, y2 = max(0.0, x1) * W, max(0.0, y1) * H, min(1.0, x2) * W, min(1.0, y2) * H
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append((int(v[0]), x1, y1, x2, y2))
    return out


def main():
    args = parse_args()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True)

    rows, hashes, report = [], set(), []
    for ds in args.sets:
        mapping = MAPPING[ds]
        root = EXT_DIR / ds
        names = yaml.safe_load(open(root / "data.yaml"))["names"]
        seen, n_img, n_dup = set(), 0, 0
        for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT):
            src = RF.sub("", path.stem)
            if src in seen:  # a Roboflow augmented copy of an image already taken
                continue
            seen.add(src)
            im = Image.open(path).convert("RGB")
            h = ahash(im)
            if h in hashes:
                n_dup += 1
                continue
            W, H = im.size
            labels = read_boxes(path.parent.parent / "labels" / (path.stem + ".txt"), W, H)
            boxes = [(mapping[names[c]], *xyxy) for c, *xyxy in labels if names[c] in mapping]
            if boxes or (ds == "traffic_count" and any(names[c] in MIXED for c, *_ in labels)):
                hashes.add(h)
            if not boxes:
                continue
            scale = min(1.0, args.max_side / max(W, H))
            if scale < 1:
                im = im.resize((round(W * scale), round(H * scale)), Image.BILINEAR)
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ds}__{src}") + ".jpg"
            im.save(img_dir / name, quality=92)
            n_img += 1
            for target, x1, y1, x2, y2 in boxes:
                rows.append(dict(image_id=name, target=target, x1=x1 * scale, y1=y1 * scale, x2=x2 * scale,
                                 y2=y2 * scale, dataset=ds, cctv=int(ds in CCTV_SETS)))
        report.append((ds, n_img, n_dup))

    for r in rows:
        r["class_id"] = CLASS_ID[r.pop("target")]

    df = pd.DataFrame(rows)
    for col in ["x1", "y1", "x2", "y2"]:
        df[col] = df[col].round(1)
    df = df[["image_id", "class_id", "x1", "y1", "x2", "y2", "dataset", "cctv"]].sort_values(
        ["dataset", "image_id"], kind="stable")
    df.to_csv(OUT_DIR / "external.csv", index=False)

    used = set(df.image_id)
    print(f"{len(used)} images, {len(df)} boxes -> {OUT_DIR}")
    print(pd.DataFrame(report, columns=["dataset", "images", "cross-set duplicates"]).to_string(index=False))
    tab = df.assign(cls=df.class_id.map(dict(enumerate(CLASS_ID)))).pivot_table(
        index="dataset", columns="cls", values="x1", aggfunc="count", fill_value=0)
    print(tab[[c for c in CLASS_ID if c in tab.columns]].to_string())


if __name__ == "__main__":
    main()
