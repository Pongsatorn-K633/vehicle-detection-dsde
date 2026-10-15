#!/usr/bin/env bash
# Version 2: assignment data + external data for the crop classifier (Kaggle private 0.696, public 0.707).
# Same detectors as version 1; only the ConvNeXt-Tiny classifier is also trained on 16,828 boxes from
# nine Roboflow datasets (sources and class mapping: EXTERNAL_DATA.md).
#
#   bash run_v1_assignment_only.sh                 # first: makes the shared detector predictions (preds/wbf_final)
#   bash run_v2_external_data.sh                   # train the classifier, then predict (~10 min on an RTX 5090)
#   bash run_v2_external_data.sh --predict-only    # use the submitted classifier in weights/
#   bash run_v2_external_data.sh --predict-only runs  # predict again with the classifier you trained (runs/)
#
# Output: outputs/v2_external_data/wbf_final_ext_a0.4.csv
# (the submitted CSV of the same name is in submissions/ and is never overwritten)
#
# Training needs either the prepared datasets/external/ (Google Drive: v2_external_prepared.zip) or the raw
# Roboflow sets unzipped into external-data/<set>/ (download list in EXTERNAL_DATA.md).
set -euo pipefail
cd "$(dirname "$0")"

CLASSES="Car Motorcycle Bus Truck Tuktuk Van Pickup Songthaew"
RECHECK="Car Truck Bus Pickup Songthaew Van"

if [[ ! -f preds/wbf_final/test.csv ]]; then
    echo "preds/wbf_final/test.csv is missing: run run_v1_assignment_only.sh first (shared detectors)" >&2
    exit 1
fi

W=runs  # classifier to predict with: freshly trained (runs/) or submitted (weights/)
if [[ "${1:-}" == "--predict-only" ]]; then
    W="${2:-weights}"
else
    if [[ -f datasets/external/external.csv ]]; then
        echo "datasets/external already exists (e.g. from Google Drive): not rebuilding"
    else
        if [[ ! -d external-data/vehicle_car ]]; then
            echo "No external data: unzip v2_external_prepared.zip (Google Drive) here, or download the Roboflow" \
                 "sets into external-data/ (EXTERNAL_DATA.md)" >&2
            exit 1
        fi
        python src/prepare_external.py
    fi
    python src/train_classifier.py --full --classes $CLASSES --samples-per-epoch 12000 \
        --external datasets/external/external.csv --name cls_all8_ext_full
fi

python src/reclassify.py --preds wbf_final --classifier $W/cls_all8_ext_full/best.pt \
    --apply-to $RECHECK --alpha 0.4 --splits test --out wbf_final_ext
python src/make_submission.py --preds wbf_final_ext_a0.4 --out outputs/v2_external_data/wbf_final_ext_a0.4.csv
