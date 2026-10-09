#!/usr/bin/env bash
# Version 1: assignment data only (Kaggle public 0.703).
# Round 2 of README section 4: RF-DETR Large + YOLO26m, flip TTA, WBF, ConvNeXt-Tiny classifier (alpha 0.4),
# all trained on the 2,991 Kaggle train images (all 15 cameras).
#
#   bash run_v1_assignment_only.sh                 # train everything, then predict (~25 min on an RTX 5090)
#   bash run_v1_assignment_only.sh --predict-only  # use the weights from Google Drive, unzipped into runs/ (~1 min)
#
# Output: submissions/v1_assignment_only/wbf_final_c8big_a0.4.csv
# The detector predictions (preds/wbf_final) are shared with run_v2_external_data.sh.
#
# Smaller GPU (see README section 3), e.g. 8 GB:
#   RFDETR_ARGS="--batch 4 --grad-accum 4 --workers 2" YOLO_ARGS="--batch 8 --workers 4" bash run_v1_assignment_only.sh
set -euo pipefail
cd "$(dirname "$0")"

CLASSES="Car Motorcycle Bus Truck Tuktuk Van Pickup Songthaew"
RECHECK="Car Truck Bus Pickup Songthaew Van"

if [[ "${1:-}" != "--predict-only" ]]; then
    if [[ -f datasets/rfdetr_full/train/_annotations.coco.json && -f datasets/yolo_full/data.yaml ]]; then
        echo "datasets/rfdetr_full and datasets/yolo_full already exist (e.g. from Google Drive): not rebuilding"
    else
        python src/prepare_data.py --full
    fi
    python src/train_rfdetr.py --data datasets/rfdetr_full --epochs 12 --name rfdl_e12_full ${RFDETR_ARGS:-}
    python src/train_yolo.py --model yolo26m.pt --data datasets/yolo_full/data.yaml --epochs 25 --patience 100 \
        --name y26m_e25_full ${YOLO_ARGS:-}
    python src/train_classifier.py --full --classes $CLASSES --samples-per-epoch 12000 --name cls_all8_full
fi

python src/predict_detector.py --weights runs/rfdl_e12_full/last_ema.pth --name rfdl_e12_full --splits test
python src/predict_detector.py --weights runs/y26m_e25_full/weights/last.pt --name y26m_e25_full --splits test
python src/fuse.py --runs rfdl_e12_full y26m_e25_full --weights 2 1 --out wbf_final --splits test
python src/reclassify.py --preds wbf_final --classifier runs/cls_all8_full/best.pt \
    --apply-to $RECHECK --alpha 0.4 --splits test --out wbf_final_c8big
python src/make_submission.py --preds wbf_final_c8big_a0.4 --out submissions/v1_assignment_only/wbf_final_c8big_a0.4.csv
