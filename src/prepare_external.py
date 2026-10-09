"""Convert the Roboflow datasets in external-data/ into one box table in our 8 classes.

    python src/prepare_external.py --classifier runs/cls_all8/best.pt

Reads external-data/<set>/{train,valid,test}/{images,labels} (YOLO export), maps each set's class names to ours
with MAPPING below (decisions and reasons: EXTERNAL_DATA.md), and writes
  datasets/external/images/<set>__<name>.jpg   one copy per source image, long side at most --max-side
  datasets/external/external.csv               image_id, class_id, x1, y1, x2, y2 (pixels), dataset, cctv, relabel

- Roboflow's augmented copies ('<name>.rf.<hash>') are kept once; images repeated across sets are kept once
  (perceptual hash, CCTV-like sets first). Polygons become their bounding box.
- RELABEL classes (mixed labels) keep the box and take the class our classifier predicts among the classes that
  label can really be (Car0: Car/Pickup/Truck, ...); relabel=1 marks them, so the crop classifier never trains on
  its own guesses.
- TUKTUK_FILTER classes keep only the boxes our classifier predicts as Tuktuk.
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from common import CLASS_ID, DATASETS_DIR, ROOT
from reclassify import load_classifier
from train_classifier import crop_box, eval_transform

EXT_DIR = ROOT / "external-data"
OUT_DIR = DATASETS_DIR / "external"
RF = re.compile(r"\.rf\.[0-9a-f]+$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

C, M, B, T, K, V, P, S = "Car", "Motorcycle", "Bus", "Truck", "Tuktuk", "Van", "Pickup", "Songthaew"
TUKTUK_FILTER = "tuktuk?"
# mixed external labels -> the classes the classifier may choose from (measured: unrestricted, it called dark
# SUVs "Bus" and white cars "Van" on this camera style)
RELABEL = {"Car0": (C, P, T), "Truck_s": (T, P, V), "Bus_s": (B, V)}
# Every name not listed is dropped. CCTV-like sets come first so cross-set duplicates keep their copy.
MAPPING = {
    "vehicle_car": {
        "sedan": C, "SUV": C, "PPV": C, "Motorcycle": M, "bus": B, "truck": T, "Solid box pick-up": T,
        "van": V, "MPV": V, "pick-up": P, "Songthaew": S},
    "traffic_count": {
        "Car1": C, "Motorcycle": M, "Bus_L": B, "Truck_L": T, "Truck_m": T, "Trailer": T, "Van": V, "Songthaew": S,
        **{k: "relabel:" + k for k in RELABEL}},
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", required=True, help="8-class crop classifier for RELABEL / TUKTUK_FILTER boxes")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--batch", type=int, default=256)
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


@torch.no_grad()
def predict_classes(rows, img_dir, model, ckpt, device, batch):
    """Classifier probabilities over our 8 class ids for each row of a box table (rows returned sorted)."""
    tf = eval_transform(ckpt["img_size"])
    ids = [CLASS_ID[c] for c in ckpt["classes"]]
    probs, crops = [], []
    rows = sorted(rows, key=lambda r: r["image_id"])
    cache = (None, None)
    for i, r in enumerate(tqdm(rows, desc="classify", leave=False)):
        if cache[0] != r["image_id"]:
            cache = (r["image_id"], Image.open(img_dir / r["image_id"]).convert("RGB"))
        crops.append(tf(crop_box(cache[1], (r["x1"], r["y1"], r["x2"], r["y2"]), ckpt["pad"])))
        if len(crops) == batch or i == len(rows) - 1:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                p = model(torch.stack(crops).to(device)).float().softmax(1).cpu().numpy()
            full = np.zeros((len(p), len(CLASS_ID)), dtype=np.float32)
            full[:, ids] = p
            probs += list(full)
            crops = []
    return rows, probs


def main():
    args = parse_args()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True)

    rows, hashes, report = [], set(), []
    for ds, mapping in MAPPING.items():
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
            boxes = [(mapping.get(names[c], None), *xyxy) for c, *xyxy in
                     read_boxes(path.parent.parent / "labels" / (path.stem + ".txt"), W, H)]
            boxes = [b for b in boxes if b[0] is not None]
            if not boxes:
                continue
            hashes.add(h)
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

    # boxes whose class comes from our classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_classifier(args.classifier, device)
    special = lambda t: t == TUKTUK_FILTER or t.startswith("relabel:")
    todo = [r for r in rows if special(r["target"])]
    done = [r for r in rows if not special(r["target"])]
    todo, probs = predict_classes(todo, img_dir, model, ckpt, device, args.batch)
    kept_tuk = 0
    for r, pr in zip(todo, probs):
        if r["target"].startswith("relabel:"):
            allowed = [CLASS_ID[c] for c in RELABEL[r["target"].split(":", 1)[1]]]
            done.append({**r, "target": None, "class_id": max(allowed, key=lambda c: pr[c]), "relabel": 1})
        elif pr.argmax() == CLASS_ID[K]:
            done.append({**r, "target": K})
            kept_tuk += 1
    for r in done:
        if r.get("target"):
            r["class_id"], r["relabel"] = CLASS_ID[r["target"]], 0

    df = pd.DataFrame(done).drop(columns="target")
    for col in ["x1", "y1", "x2", "y2"]:
        df[col] = df[col].round(1)
    df = df[["image_id", "class_id", "x1", "y1", "x2", "y2", "dataset", "cctv", "relabel"]].sort_values(
        ["dataset", "image_id"], kind="stable")
    df.to_csv(OUT_DIR / "external.csv", index=False)

    # images left without any box after filtering are removed
    used = set(df.image_id)
    for p in img_dir.iterdir():
        if p.name not in used:
            p.unlink()

    print(f"{len(used)} images, {len(df)} boxes -> {OUT_DIR}")
    print(pd.DataFrame(report, columns=["dataset", "images", "cross-set duplicates"]).to_string(index=False))
    print(f"3 wheel motorbike kept as Tuktuk: {kept_tuk} of {sum(r['target'] == TUKTUK_FILTER for r in todo)}")
    tab = df.assign(cls=df.class_id.map(dict(enumerate(CLASS_ID)))).pivot_table(
        index="dataset", columns="cls", values="x1", aggfunc="count", fill_value=0)
    print(tab[[c for c in CLASS_ID if c in tab.columns]].to_string())
    print("re-labelled boxes:", int(df.relabel.sum()))


if __name__ == "__main__":
    main()
