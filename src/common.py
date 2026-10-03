"""Shared paths, class names and helpers for the Bangkok CCTV vehicle detection pipeline."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "original-data"
DOCS_DIR = ROOT / "docs"
TRAIN_IMG_DIR = DATA_DIR / "train" / "train"
TEST_IMG_DIR = DATA_DIR / "test" / "test"
TRAIN_CSV = DOCS_DIR / "train.csv"
SAMPLE_SUB_CSV = DOCS_DIR / "sample_submission.csv"

YOLO_DIR = ROOT / "datasets" / "yolo"
RUNS_DIR = ROOT / "runs"
SUBMISSION_DIR = ROOT / "submissions"

CLASS_NAMES = ["Car", "Motorcycle", "Bus", "Truck", "Tuktuk", "Van", "Pickup", "Songthaew"]

# Test cameras (1068, 1072, 1192, 1439, 227) never appear in train, so validation
# holds out whole cameras. These three keep some of every rare class in val
# (Songthaew 24, Tuktuk 44, Van 31) while leaving camera 1426 (most Songthaew) in train.
VAL_CAMERAS = ["1066", "1427", "172"]

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


def predict_images(model, paths, imgsz, conf, max_det, augment=False, batch=32, device=None):
    """Run a YOLO model over image paths.

    Yields (file_name, xyxy[N,4], scores[N], classes[N]) as numpy arrays in original pixel coords.
    """
    paths = [str(p) for p in paths]
    for i in range(0, len(paths), batch):
        results = model.predict(
            paths[i:i + batch], imgsz=imgsz, conf=conf, max_det=max_det,
            augment=augment, device=device, verbose=False,
        )
        for r in results:
            b = r.boxes
            yield (
                Path(r.path).name,
                b.xyxy.cpu().numpy(),
                b.conf.cpu().numpy(),
                b.cls.cpu().numpy().astype(int),
            )
