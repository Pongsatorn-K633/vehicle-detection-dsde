# Bangkok CCTV Vehicle Detection

2110531 Data Science and Data Engineering Tools, take-home midterm (2026/1).
8-class object detection on BMA traffic camera images, scored by mAP@50 (pycocotools) on Kaggle.

## Setup

```
conda env create -f environment.yml
conda activate vehicle-det
```

Put the Kaggle images in place (they are not in git):

```
docs/
  train.csv               (in git)
  sample_submission.csv   (in git)
original-data/
  train/train/*.jpg       (2,991 images)
  test/test/*.jpg         (1,013 images)
```

## Pipeline

Run from the repo root.

```
# 1. YOLO dataset with whole cameras held out for validation (test cameras are unseen in train)
python src/prepare_data.py

# 2. Train
python src/train.py --name y26m_640

# 3. Local mAP@50 on the validation cameras (per class, class-agnostic, conf sweep)
python src/evaluate.py --weights runs/y26m_640/weights/best.pt

# 4. Final model on all cameras, then the submission
python src/prepare_data.py --full
python src/train.py --data datasets/yolo_full/data.yaml --name y26m_640_full
python src/predict.py --weights runs/y26m_640_full/weights/best.pt
```

Submissions are written to `submissions/<run name>.csv`.

## Notes

- Validation cameras: 1066, 1427, 172 (see `VAL_CAMERAS` in `src/common.py`).
- Predictions use `conf=0.001`, `max_det=100`: low-confidence boxes rank last and cannot lower AP,
  and pycocotools only counts the top 100 detections per image.
- Submission `image_id`s are the long names from `sample_submission.csv`, matched to test files by camera and timestamp.
