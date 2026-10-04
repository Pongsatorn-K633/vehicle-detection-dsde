"""Turn test predictions into a Kaggle submission CSV.

    python src/make_submission.py --preds wbf_final_cls_a0.5        # -> submissions/wbf_final_cls_a0.5.csv

image_id is the test file name ('<cam>_<date>_<time>.jpg', the same style as train.csv).
sample_submission.csv shows long Thai ids, but Kaggle's scorer only matches the short file names:
a file with the long ids scores exactly 0. Only the images listed in sample_submission.csv are
written (16 test images are not scored). At most --max-det boxes per image.
"""
import argparse

from common import ROOT, SUBMISSION_DIR, cap_per_image, read_preds, test_long_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="prediction folder under preds/")
    p.add_argument("--out", default=None, help="default submissions/<preds>.csv")
    p.add_argument("--max-det", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    out = ROOT / args.out if args.out else SUBMISSION_DIR / f"{args.preds}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    long_id = test_long_ids()
    df = read_preds(args.preds, "test")
    missing = set(long_id) - set(df.image)
    df = cap_per_image(df[df.image.isin(long_id)], args.max_det)

    sub = df.rename(columns={"image": "image_id"})[["image_id", "class_id", "score", "x1", "y1", "x2", "y2"]]
    sub = sub.rename(columns={"score": "confidence"}).round({"confidence": 5, "x1": 2, "y1": 2, "x2": 2, "y2": 2})
    sub.insert(0, "id", range(len(sub)))
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(sub)} boxes for {sub.image_id.nunique()}/{len(long_id)} images -> {out}")
    if missing:
        print(f"Note: {len(missing)} images have no box at all (fine if the model found nothing there)")
    print(sub.class_id.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
