# Bangkok CCTV Vehicle Detection

Take-home midterm for 2110531 Data Science and Data Engineering Tools (2026/1).
The task is 8-class vehicle detection on Bangkok (BMA) traffic camera images, scored on Kaggle with **mAP@50** (pycocotools).

## Contents

1. [Dataset](#1-dataset)
2. [Setup](#2-setup)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [Inference (Kaggle submission)](#5-inference-kaggle-submission)
6. [Baseline configuration](#6-baseline-configuration)
7. [Project structure](#7-project-structure)

## 1. Dataset

| Item | Value |
|---|---|
| Source | BMA Traffic CCTV, one frame every 30 minutes, 25–31 Aug 2026, 06:00–21:00 |
| Image size | 352 × 288 px (all images) |
| Classes | 8 |
| Train images | 2,991 (2,889 with boxes, 102 without any box, used as background examples) |
| Test images | 1,013 (997 are listed in `sample_submission.csv`) |
| Labelled boxes | 27,396 (average 9.5 per image, maximum 48) |
| Cameras | 15 in train, 5 different ones in test (1068, 1072, 1192, 1439, 227) |

**Train / validation / test split.** Validation is taken from the labelled training images by holding out whole cameras (**1066, 1427, 172**), because the test set uses cameras that never appear in train.

| ID | Class | All train boxes | Train split | Val split | Test* |
|---|---|---|---|---|---|
| 0 | Car | 17,762 | 15,282 | 2,480 | – |
| 1 | Motorcycle | 6,563 | 5,172 | 1,391 | – |
| 2 | Bus | 482 | 415 | 67 | – |
| 3 | Truck | 1,356 | 1,072 | 284 | – |
| 4 | Tuktuk | 265 | 221 | 44 | – |
| 5 | Van | 354 | 323 | 31 | – |
| 6 | Pickup | 497 | 383 | 114 | – |
| 7 | Songthaew | 117 | 93 | 24 | – |
| | **Images** | **2,991** | **2,388** | **603** | **1,013** |
| | **Cameras** | **15** | **12** | **3** | **5** |

\* **The test set has no labels, so it cannot be evaluated locally.** It is used only to create the submission and is scored on Kaggle against hidden labels (997 of its 1,013 images are scored). The validation split is the local stand-in for the Kaggle score.

Things to know about the data:
- **Objects are very small.** 80% of cars and 97% of motorcycles are under 32 × 32 px.
- **Classes are very unbalanced.** Car and Motorcycle make up 89% of boxes, but mAP@50 averages all 8 classes equally.
- **Songthaew comes mostly from one camera.** 91 of its 117 boxes are from camera 1426.

## 2. Setup

**1. Create the environment** (Miniconda):

```
conda env create -f environment.yml
conda activate vehicle-det
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

PyTorch is installed from the CUDA 12.6 wheels, which run on both Pascal (GTX 10xx) and newer GPUs.

**2. Data.** The images and CSVs are included in the repo, so cloning is enough:

```
docs/
  train.csv               labels
  sample_submission.csv   submission template
original-data/
  train/train/*.jpg       2,991 images
  test/test/*.jpg         1,013 images
```

## 3. Training

Run every command from the repo root. There are two kinds of training runs:

| Run | Trained on | Used for |
|---|---|---|
| **Experiment** | 12 cameras (val cameras held out) | Measuring changes locally ([Evaluation](#4-evaluation)) |
| **Final** | All 15 cameras | The Kaggle submission ([Inference](#5-inference-kaggle-submission)) |

The final run uses the same settings, but it also learns from the 3 validation cameras, which include 24 more Songthaew and 44 more Tuktuk boxes.

### 3.1 Experiment model

```
python src/prepare_data.py          # original-data/ → datasets/yolo/       (12 train cameras + 3 val)
python src/train.py --name y26m_640
```

### 3.2 Final model

```
python src/prepare_data.py --full   # original-data/ → datasets/yolo_full/  (all 15 cameras in train)
python src/train.py --data datasets/yolo_full/data.yaml --name y26m_640_full
```

If the GPU runs out of memory, add `--batch 8`. Weights are saved to `runs/<name>/weights/best.pt`.

## 4. Evaluation

Scores the experiment model on the 3 validation cameras with the same metric Kaggle uses:

```
python src/evaluate.py --weights runs/y26m_640/weights/best.pt
```

The output includes:
- **mAP50:** the 8-class mean (the Kaggle metric).
- **Per-class AP50.**
- **agnostic_AP50:** class ignored, so it measures box quality only. A large gap between this and mAP50 means the boxes are right but the classes are wrong.
- **A sweep over confidence thresholds** 0.001 / 0.01 / 0.05 / 0.25.

Results are saved to `runs/<name>/eval/ap50.csv`.

## 5. Inference (Kaggle submission)

Predict the test images with the final model (section 3.2):

```
python src/predict.py --weights runs/y26m_640_full/weights/best.pt
```

This writes `submissions/y26m_640_full.csv`. Upload that file to Kaggle. Any other weights work too, for example the experiment model for a quick first submission.

How `predict.py` builds the file:
- It uses `conf=0.001` and `max_det=100`. Low-confidence boxes are ranked last, so they cannot lower AP, and pycocotools only counts the top 100 boxes per image.
- The `image_id`s are the long names from `sample_submission.csv`, matched to test files by camera and timestamp.

## 6. Baseline configuration

| Setting | Value |
|---|---|
| Model | **YOLO26m** (Ultralytics, COCO-pretrained `yolo26m.pt`) |
| Input size | 640 (images are enlarged about 1.8× to help with small objects) |
| Epochs | 100, early stop after 30 epochs with no improvement |
| Batch | 16 |
| Learning rate | Ultralytics defaults, cosine schedule (`cos_lr=True`) |
| Augmentation | Ultralytics defaults (mosaic, horizontal flip, HSV, scale); mosaic turned off for the last 10 epochs |
| Copy-paste | Off |
| Reproducibility | `seed=0`, `deterministic=True` |
| Inference | `imgsz=640`, `conf=0.001`, `max_det=100`, no test-time augmentation |

Every run's full settings are saved to `runs/<name>/args.yaml`.

## 7. Project structure

```
docs/               train.csv, sample_submission.csv
original-data/      Kaggle train and test images
src/
  common.py         paths, class names, validation cameras, helpers
  prepare_data.py   CSV -> YOLO dataset + validation ground truth
  train.py          training
  evaluate.py       local mAP@50
  predict.py        Kaggle submission CSV
datasets/           generated by prepare_data.py (not in git)
runs/               training outputs and weights (not in git)
submissions/        submission CSVs
environment.yml     conda environment
```
