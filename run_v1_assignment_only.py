"""Version 1: assignment data only (Kaggle private 0.679, public 0.703).

Python version of run_v1_assignment_only.sh, with the same steps and options, for Windows, macOS and Linux.
Round 2 of README section 4: RF-DETR Large + YOLO26m, flip TTA, WBF, ConvNeXt-Tiny classifier (alpha 0.4),
all trained on the 2,991 Kaggle train images (all 15 cameras).

    python run_v1_assignment_only.py                      # train everything, then predict (~25 min on an RTX 5090)
    python run_v1_assignment_only.py --predict-only       # use the submitted weights in weights/ (~1 min)
    python run_v1_assignment_only.py --predict-only runs  # predict again with weights you trained (runs/)

Training writes to runs/ and predicts with those new weights; weights/ (the submitted ones) is never touched.

Output: outputs/v1_assignment_only/wbf_final_c8big_a0.4.csv
(the submitted CSV of the same name is in submissions/ and is never overwritten)
The detector predictions (preds/wbf_final) are shared with run_v2_external_data.py.

Smaller GPU (see README section 3), e.g. 8 GB:
    python run_v1_assignment_only.py --rfdetr-args "--batch 4 --grad-accum 4 --workers 2" --yolo-args "--batch 8 --workers 4"
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = "Car Motorcycle Bus Truck Tuktuk Van Pickup Songthaew".split()
RECHECK = "Car Truck Bus Pickup Songthaew Van".split()


def run(*args):
    """Run a src/ script with this Python; stop on the first failure, like `set -e` in the .sh."""
    print("+ python " + " ".join(args), flush=True)
    code = subprocess.run([sys.executable, *args], cwd=ROOT).returncode
    if code:
        sys.exit(code)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predict-only", nargs="?", const="weights", default=None, metavar="WEIGHTS_DIR",
                   help="skip training; predict with weights/ (default) or the given folder, e.g. runs")
    p.add_argument("--rfdetr-args", default=os.environ.get("RFDETR_ARGS", ""), help="extra train_rfdetr.py flags")
    p.add_argument("--yolo-args", default=os.environ.get("YOLO_ARGS", ""), help="extra train_yolo.py flags")
    return p.parse_args()


def main():
    args = parse_args()
    w = args.predict_only or "runs"  # weights to predict with: freshly trained (runs/) or submitted (weights/)
    if args.predict_only is None:
        if (ROOT / "datasets/rfdetr_full/train/_annotations.coco.json").is_file() and \
                (ROOT / "datasets/yolo_full/data.yaml").is_file():
            print("datasets/rfdetr_full and datasets/yolo_full already exist (e.g. from Google Drive): not rebuilding")
        else:
            run("src/prepare_data.py", "--full")
        run("src/train_rfdetr.py", "--data", "datasets/rfdetr_full", "--epochs", "12", "--name", "rfdl_e12_full",
            *shlex.split(args.rfdetr_args))
        run("src/train_yolo.py", "--model", "yolo26m.pt", "--data", "datasets/yolo_full/data.yaml", "--epochs", "25",
            "--patience", "100", "--name", "y26m_e25_full", *shlex.split(args.yolo_args))
        run("src/train_classifier.py", "--full", "--classes", *CLASSES, "--samples-per-epoch", "12000",
            "--name", "cls_all8_full")

    run("src/predict_detector.py", "--weights", f"{w}/rfdl_e12_full/last_ema.pth", "--name", "rfdl_e12_full",
        "--splits", "test")
    run("src/predict_detector.py", "--weights", f"{w}/y26m_e25_full/weights/last.pt", "--name", "y26m_e25_full",
        "--splits", "test")
    run("src/fuse.py", "--runs", "rfdl_e12_full", "y26m_e25_full", "--weights", "2", "1", "--out", "wbf_final",
        "--splits", "test")
    run("src/reclassify.py", "--preds", "wbf_final", "--classifier", f"{w}/cls_all8_full/best.pt",
        "--apply-to", *RECHECK, "--alpha", "0.4", "--splits", "test", "--out", "wbf_final_c8big")
    run("src/make_submission.py", "--preds", "wbf_final_c8big_a0.4",
        "--out", "outputs/v1_assignment_only/wbf_final_c8big_a0.4.csv")


if __name__ == "__main__":
    main()
