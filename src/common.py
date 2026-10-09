"""Shared paths, class names and helpers for the Bangkok CCTV vehicle detection pipeline."""
from pathlib import Path
import contextlib
import io
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "original-data"  # the Kaggle download
TRAIN_IMG_DIR = DATA_DIR / "train" / "train"
TEST_IMG_DIR = DATA_DIR / "test" / "test"
TRAIN_CSV = DATA_DIR / "train.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"

DATASETS_DIR = ROOT / "datasets"
YOLO_DIR = DATASETS_DIR / "yolo"
RUNS_DIR = ROOT / "runs"
PREDS_DIR = ROOT / "preds"
SUBMISSION_DIR = ROOT / "submissions"

CLASS_NAMES = ["Car", "Motorcycle", "Bus", "Truck", "Tuktuk", "Van", "Pickup", "Songthaew"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Test cameras (1068, 1072, 1192, 1439, 227) never appear in train, so validation
# holds out whole cameras. These three keep some of every rare class in val
# (Songthaew 24, Tuktuk 44, Van 31) while leaving camera 1426 (most Songthaew) in train.
VAL_CAMERAS = ["1066", "1427", "172"]

# Columns of every prediction file in preds/<run>/<split>.csv (pixel xyxy).
PRED_COLUMNS = ["image", "class_id", "score", "x1", "y1", "x2", "y2"]

_TIMESTAMP = re.compile(r"(\d{8}_\d{6})\.jpg$")


def camera_of(fname: str) -> str:
    """'1066_20260825_060110.jpg' -> '1066'."""
    return fname.split("_")[0]


def short_test_name(long_id: str) -> str:
    """Map a submission image_id to its test file name.

    'dataset_1068_annotated_coco1.0_1068_<thai>_20260825_060110.jpg' -> '1068_20260825_060110.jpg'
    """
    cam = long_id.split("_")[1]
    ts = _TIMESTAMP.search(long_id).group(1)
    return f"{cam}_{ts}.jpg"


# ---------------------------------------------------------------- image lists

def val_images(val_cameras=VAL_CAMERAS):
    """Sorted file names of the labelled images from the validation cameras (incl. images without boxes)."""
    cams = set(val_cameras)
    return sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpg") if camera_of(p.name) in cams)


def test_long_ids():
    """{short test file name: long submission image_id} for the images Kaggle scores."""
    sample = pd.read_csv(SAMPLE_SUB_CSV, encoding="utf-8")
    return {short_test_name(i): i for i in sample.image_id.unique()}


def split_images(split):
    """(image_dir, file names) for split 'val' or 'test'."""
    if split == "val":
        return TRAIN_IMG_DIR, val_images()
    if split == "test":
        return TEST_IMG_DIR, sorted(test_long_ids())
    raise ValueError(f"unknown split {split!r}")


def image_dir_of(fname: str) -> Path:
    """Folder that holds an image (train and test file names never overlap)."""
    return TRAIN_IMG_DIR if (TRAIN_IMG_DIR / fname).exists() else TEST_IMG_DIR


def load_gt(clip=True):
    """train.csv with boxes clipped to the image (all images are 352x288) and empty boxes removed."""
    df = pd.read_csv(TRAIN_CSV)
    if clip:
        df[["x1", "x2"]] = df[["x1", "x2"]].clip(0, 352)
        df[["y1", "y2"]] = df[["y1", "y2"]].clip(0, 288)
        df = df[(df.x2 > df.x1) & (df.y2 > df.y1)]
    return df


# ---------------------------------------------------------------- prediction files

def pred_path(run: str, split: str, flip=False) -> Path:
    """preds/<run>/<split>.csv, or <split>_flip.csv for predictions on the mirrored image."""
    p = Path(run)
    base = p if p.is_absolute() or p.parts[0] == "preds" else PREDS_DIR / run
    return (ROOT / base) / f"{split}{'_flip' if flip else ''}.csv"


def read_preds(run: str, split: str, flip=False) -> pd.DataFrame:
    return pd.read_csv(pred_path(run, split, flip))


def write_preds(df: pd.DataFrame, run: str, split: str, flip=False) -> Path:
    out = pred_path(run, split, flip)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df[PRED_COLUMNS].copy()
    df[["score", "x1", "y1", "x2", "y2"]] = df[["score", "x1", "y1", "x2", "y2"]].round(5)
    df.to_csv(out, index=False)
    return out


def cap_per_image(df: pd.DataFrame, max_det: int) -> pd.DataFrame:
    """Keep the max_det highest-scoring boxes of each image (pycocotools ignores the rest anyway)."""
    df = df.sort_values(["image", "score"], ascending=[True, False])
    return df[df.groupby("image").cumcount() < max_det].reset_index(drop=True)


# ---------------------------------------------------------------- scoring (same as Kaggle)

def build_val_coco(images):
    """pycocotools ground truth for the given labelled images. Returns (COCO, {file name: image id})."""
    from pycocotools.coco import COCO

    gt = load_gt()
    gt = gt[gt.image_id.isin(set(images))]
    id_of = {f: i for i, f in enumerate(images)}
    data = {
        "images": [{"id": i, "file_name": f, "width": 352, "height": 288} for f, i in id_of.items()],
        "annotations": [
            {"id": k, "image_id": id_of[r.image_id], "category_id": int(r.class_id),
             "bbox": [float(r.x1), float(r.y1), float(r.x2 - r.x1), float(r.y2 - r.y1)],
             "area": float((r.x2 - r.x1) * (r.y2 - r.y1)), "iscrowd": 0}
            for k, r in enumerate(gt.itertuples())
        ],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)],
    }
    coco = COCO()
    coco.dataset = data
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, id_of


def coco_ap50(coco, id_of, preds: pd.DataFrame, use_cats=True):
    """Return (mAP50, per-class AP50 list) with pycocotools. Classes without ground truth are NaN and skipped."""
    from pycocotools.cocoeval import COCOeval

    k = len(CLASS_NAMES) if use_cats else 1
    preds = preds[preds.image.isin(id_of)]
    if preds.empty:
        return 0.0, [0.0] * k
    dets = [{"image_id": id_of[r.image], "category_id": int(r.class_id),
             "bbox": [float(r.x1), float(r.y1), float(r.x2 - r.x1), float(r.y2 - r.y1)], "score": float(r.score)}
            for r in preds.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.useCats = int(use_cats)
        ev.evaluate()
        ev.accumulate()
    # precision[T, R, K, A, M]: IoU=0.50, all recall points, class k, area=all, maxDets=100
    prec = ev.eval["precision"][0, :, :, 0, -1]
    per_class = [float(prec[:, c][prec[:, c] > -1].mean()) if (prec[:, c] > -1).any() else float("nan")
                 for c in range(prec.shape[1])]
    return float(np.nanmean(per_class)), per_class


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between xyxy boxes a[N,4] and b[M,4]."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
