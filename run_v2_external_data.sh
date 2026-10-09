#!/usr/bin/env bash
# Version 2: assignment data + external data for the crop classifier (Kaggle public 0.707).
# Same detectors as version 1; only the ConvNeXt-Tiny classifier is also trained on 18,867 boxes from
# nine Roboflow datasets (sources and class mapping: docs/external_data.md).
#
#   bash run_v1_assignment_only.sh                 # first: makes the shared detector predictions (preds/wbf_final)
#   bash run_v2_external_data.sh                   # train the classifier, then predict (~10 min on an RTX 5090)
#   bash run_v2_external_data.sh --predict-only    # use the classifier from Google Drive, unzipped into runs/
#
# Output: submissions/v2_external_data/wbf_final_ext_a0.4.csv
#
# Training needs either the prepared datasets/external/ (Google Drive) or the raw Roboflow sets unzipped into
# external-data/<set>/ (Google Drive, or downloaded from the sources in docs/external_data.md).
set -euo pipefail
cd "$(dirname "$0")"

CLASSES="Car Motorcycle Bus Truck Tuktuk Van Pickup Songthaew"
RECHECK="Car Truck Bus Pickup Songthaew Van"

if [[ ! -f preds/wbf_final/test.csv ]]; then
    echo "preds/wbf_final/test.csv is missing: run run_v1_assignment_only.sh first (shared detectors)" >&2
    exit 1
fi

if [[ "${1:-}" != "--predict-only" ]]; then
    if [[ -f datasets/external/external.csv ]]; then
        echo "datasets/external already exists (e.g. from Google Drive): not rebuilding"
    else
        # round-1 classifier (validation cameras held out); only used to relabel mixed external labels
        python src/train_classifier.py --classes $CLASSES --samples-per-epoch 12000 --name cls_all8
        python src/prepare_external.py --classifier runs/cls_all8/best.pt
    fi
    python src/train_classifier.py --full --classes $CLASSES --samples-per-epoch 12000 \
        --external datasets/external/external.csv --name cls_all8_ext_full
fi

python src/reclassify.py --preds wbf_final --classifier runs/cls_all8_ext_full/best.pt \
    --apply-to $RECHECK --alpha 0.4 --splits test --out wbf_final_ext
python src/make_submission.py --preds wbf_final_ext_a0.4 --out submissions/v2_external_data/wbf_final_ext_a0.4.csv
