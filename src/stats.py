"""Dataset statistics of report section 2.1 (Table 2.1 and the observations under it).

    python src/stats.py

Boxes are read with load_gt(), i.e. clipped to the image and empty boxes removed, the same as for training.
"Small" follows the COCO definition: box area under 32 x 32 px.
"""
import pandas as pd
from PIL import Image

from common import CLASS_NAMES, SAMPLE_SUB_CSV, TEST_IMG_DIR, TRAIN_IMG_DIR, VAL_CAMERAS, camera_of, load_gt


def main():
    df = load_gt()
    df["camera"] = df.image_id.map(camera_of)
    df["w"], df["h"] = df.x2 - df.x1, df.y2 - df.y1
    df["cls"] = df.class_id.map(dict(enumerate(CLASS_NAMES)))

    train_files = sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpg"))
    test_files = sorted(p.name for p in TEST_IMG_DIR.glob("*.jpg"))
    sizes = {Image.open(d / f).size for d, fs in [(TRAIN_IMG_DIR, train_files), (TEST_IMG_DIR, test_files)] for f in fs}
    labelled = set(df.image_id) & set(train_files)
    per_img = df.groupby("image_id").size()
    train_cams = sorted({camera_of(f) for f in train_files}, key=int)
    test_cams = sorted({camera_of(f) for f in test_files}, key=int)

    print(f"Image sizes (w x h): {sorted(sizes)}")
    print(f"Train images: {len(train_files):,} ({len(labelled):,} with boxes, {len(train_files) - len(labelled):,} without)")
    print(f"Test images: {len(test_files):,} ({pd.read_csv(SAMPLE_SUB_CSV).image_id.nunique():,} in sample_submission.csv)")
    print(f"Labelled boxes: {len(df):,} (average {per_img.mean():.1f} per labelled image, maximum {per_img.max()})")
    print(f"Train cameras ({len(train_cams)}): {train_cams}")
    print(f"Test cameras ({len(test_cams)}): {test_cams}; shared with train: {sorted(set(train_cams) & set(test_cams))}")

    is_val = df.camera.isin(VAL_CAMERAS)
    table = pd.DataFrame({"all": df.cls.value_counts(), "train split": df[~is_val].cls.value_counts(),
                          "val split": df[is_val].cls.value_counts()}).reindex(CLASS_NAMES).fillna(0).astype(int)
    val_files = [f for f in train_files if camera_of(f) in VAL_CAMERAS]
    table.loc["images"] = [len(train_files), len(train_files) - len(val_files), len(val_files)]
    print(f"\nTable 2.1 (val cameras {VAL_CAMERAS}):\n{table.to_string()}")

    share = table.loc[["Car", "Motorcycle"], "all"].sum() / len(df)
    print(f"\nCar + Motorcycle: {share:.0%} of all boxes")
    small = (df.w * df.h < 32 * 32).groupby(df.cls).mean().reindex(CLASS_NAMES)
    print("Share of boxes under 32 x 32 px (area):\n" + small.map("{:.0%}".format).to_string())
    med = df.groupby("cls")[["w", "h"]].median().reindex(CLASS_NAMES)
    print("Median box size w x h (px):\n" + med.apply(lambda r: f"{r.w:g} x {r.h:g}", axis=1).to_string())

    s = df[df.cls == "Songthaew"].camera.value_counts()
    print(f"\nSongthaew boxes per camera ({s.sum()} in total):\n{s.to_string()}")


if __name__ == "__main__":
    main()
