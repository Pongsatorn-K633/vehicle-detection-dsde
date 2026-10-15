"""Version 2: assignment data + external data for the crop classifier (Kaggle private 0.696, public 0.707).

Python version of run_v2_external_data.sh, with the same steps and options, for Windows, macOS and Linux.
Same detectors as version 1; only the ConvNeXt-Tiny classifier is also trained on 16,828 boxes from
nine Roboflow datasets (sources and class mapping: EXTERNAL_DATA.md).

    python run_v1_assignment_only.py                      # first: makes the shared detector predictions (preds/wbf_final)
    python run_v2_external_data.py                        # train the classifier, then predict (~10 min on an RTX 5090)
    python run_v2_external_data.py --predict-only         # use the submitted classifier in weights/
    python run_v2_external_data.py --predict-only runs    # predict again with the classifier you trained (runs/)

Output: outputs/v2_external_data/wbf_final_ext_a0.4.csv
(the submitted CSV of the same name is in submissions/ and is never overwritten)

Training needs either the prepared datasets/external/ (Google Drive: v2_external_prepared.zip) or the raw
Roboflow sets unzipped into external-data/<set>/ (download list in EXTERNAL_DATA.md).
"""
import argparse
import sys

from run_v1_assignment_only import CLASSES, RECHECK, ROOT, run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predict-only", nargs="?", const="weights", default=None, metavar="WEIGHTS_DIR",
                   help="skip training; predict with weights/ (default) or the given folder, e.g. runs")
    return p.parse_args()


def main():
    args = parse_args()
    if not (ROOT / "preds/wbf_final/test.csv").is_file():
        sys.exit("preds/wbf_final/test.csv is missing: run run_v1_assignment_only.py first (shared detectors)")

    w = args.predict_only or "runs"  # classifier to predict with: freshly trained (runs/) or submitted (weights/)
    if args.predict_only is None:
        if (ROOT / "datasets/external/external.csv").is_file():
            print("datasets/external already exists (e.g. from Google Drive): not rebuilding")
        else:
            if not (ROOT / "external-data/vehicle_car").is_dir():
                sys.exit("No external data: unzip v2_external_prepared.zip (Google Drive) here, or download the "
                         "Roboflow sets into external-data/ (EXTERNAL_DATA.md)")
            run("src/prepare_external.py")
        run("src/train_classifier.py", "--full", "--classes", *CLASSES, "--samples-per-epoch", "12000",
            "--external", "datasets/external/external.csv", "--name", "cls_all8_ext_full")

    run("src/reclassify.py", "--preds", "wbf_final", "--classifier", f"{w}/cls_all8_ext_full/best.pt",
        "--apply-to", *RECHECK, "--alpha", "0.4", "--splits", "test", "--out", "wbf_final_ext")
    run("src/make_submission.py", "--preds", "wbf_final_ext_a0.4",
        "--out", "outputs/v2_external_data/wbf_final_ext_a0.4.csv")


if __name__ == "__main__":
    main()
