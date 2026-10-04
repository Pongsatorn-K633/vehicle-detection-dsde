# Bangkok CCTV Vehicle Detection: Training Report

As of 2026-10-04. Round 1 (experiment models) and Round 2 (final models) are both finished.

**Result:** 0.703 mAP@50 on the Kaggle public leaderboard (1st place) with the Round 2 models. The Round 1
models scored 0.637 public and 0.693 on the local validation set. Full marks for the mAP part of the grade
start at 0.56.

## 1. Data handling

### 1.1 Train / validation split by camera

The test set uses 5 cameras that never appear in train, so validation must also use unseen cameras.
Three whole cameras (**1066, 1427, 172**) are held out; a random image split would put near-identical
frames of the same camera in train and val and overstate the score.

| Split | Cameras | Images | Boxes |
|---|---|---|---|
| Train (Round 1) | 12 | 2,388 | 22,961 |
| Validation | 3 (1066, 1427, 172) | 603 | 4,435 |
| Train (Round 2, `--full`) | 15 | 2,991 | 27,396 |
| Test (Kaggle) | 5 (1068, 1072, 1192, 1439, 227) | 1,013 (997 scored) | hidden |

102 train images have no box at all. They are kept as background images, so the models learn what an
empty road looks like.

Box coordinates are clipped to the 352 × 288 image, and empty boxes are removed (`load_gt` in `common.py`).

### 1.2 Class re-balancing: duplicating images (oversampling)

Car and Motorcycle are 89% of all boxes, but mAP@50 weights all 8 classes equally. `prepare_data.py`
therefore copies training images that contain a rare class (repeat-factor sampling, from the LVIS paper,
Gupta et al. 2019):

- For each class c: r_c = max(1, sqrt(t / f_c)), where f_c is the fraction of train images that contain c
  and t = 0.3 (`--repeat-thr`).
- Each image is copied round(largest r_c of its classes) times, at most 4. Copies are saved as
  `<name>__rep<k>.jpg`.
- **Every box in a copied image stays labelled**, so no vehicle is ever turned into background.
- Validation images are never copied, so the validation score stays honest.
- Copies are not identical in training: each copy gets its own random augmentation (section 1.3).

Effect on the Round 1 training set (2,388 → 3,178 images):

| Class | Boxes before | Boxes after | × |
|---|---|---|---|
| Car | 15,282 | 21,075 | 1.4 |
| Motorcycle | 5,172 | 6,922 | 1.3 |
| Bus | 415 | 564 | 1.4 |
| Truck | 1,072 | 1,465 | 1.4 |
| Tuktuk | 221 | 442 | 2.0 |
| Van | 323 | 648 | 2.0 |
| Pickup | 383 | 774 | 2.0 |
| Songthaew | 93 | 372 | 4.0 |

Common classes grow a little too, because they appear in the same images as the rare ones.

### 1.3 Augmentation

No rotation and no vertical flip in any model: the cameras are fixed and vehicles are never upside down.

| Model | Augmentation |
|---|---|
| RF-DETR | horizontal flip (p 0.5); brightness/contrast ±20% (p 0.5); hue ±5, saturation ±20, value ±15 (p 0.3) for day, night and rain; multi-scale training around 704 px |
| YOLO26 | mosaic (4 images in one, switched off for the last 10 epochs); horizontal flip (p 0.5); HSV jitter (hue 0.015, saturation 0.7, value 0.4); shift ±10%; zoom ±30% (milder than the default ±50%, because objects are already tiny); random erasing 0.4 |
| Classifier | each box randomly shifted up to 10% and resized 85–120%, so it also learns from slightly-off detector boxes; horizontal flip; colour jitter 0.3; label smoothing 0.1 |

### 1.4 Image size

All images are 352 × 288 px and objects are very small: 80% of cars and 97% of motorcycles are under
32 × 32 px. Both detectors therefore train and predict at **704 px** (images enlarged 2×).

### 1.5 Data for the classifier

The classifier is trained on box crops cut from `train.csv` (ground-truth boxes, plus 15% context on each
side, resized to 224 × 224). Rare classes are drawn more often: sampling weight = count^-0.5, 12,000 crops
per epoch.

## 2. Pipeline

1. **RF-DETR Large** (transformer detector, DINOv2 backbone, COCO-pretrained) predicts boxes.
2. **YOLO26** (CNN detector, COCO-pretrained) predicts boxes. It makes different mistakes from RF-DETR.
3. **Flip TTA:** each detector also predicts on the mirrored image; those boxes are mirrored back.
4. **Weighted Boxes Fusion (WBF):** the 4 prediction sets (2 models × normal/mirrored) are merged into one,
   RF-DETR weighted 2, YOLO 1. Each model's own duplicate boxes are removed first (NMS).
5. **ConvNeXt-Tiny classifier:** re-checks boxes labelled Car, Truck, Bus, Pickup, Songthaew or Van and adds
   a second, lower-scored guess when it disagrees (details in 3.4).

Prediction keeps very low-scored boxes (score ≥ 0.001), at most 100 per image: mAP ranks boxes by score,
so extra low boxes can only add recall.

## 3. Round 1: experiment models (12 cameras, validated on 3)

### 3.1 Training curves

| Model | Epochs | Best epoch | Best val mAP50 | Last epoch val mAP50 |
|---|---|---|---|---|
| RF-DETR Large | 40 | 6 | 0.635 | 0.577 |
| YOLO26l | 60 (early stop at 38) | 35 | 0.604 | 0.592 |

RF-DETR peaked at epoch 6 and then declined while training loss kept falling: it was learning the 12
training cameras rather than skills that carry over to new cameras. The best-epoch checkpoint was kept.
(Values from each library's own validation; section 3.2 uses the Kaggle-style pycocotools score.)

### 3.2 What each stage adds (validation, pycocotools mAP@50)

| Stage | mAP50 | Change |
|---|---|---|
| RF-DETR alone | 0.634 | |
| + remove own duplicates | 0.639 | +0.005 |
| + flip TTA | 0.655 | +0.016 |
| + YOLO26l, WBF 1:1 | 0.676 | +0.021 |
| + YOLO26l, WBF 2:1 | 0.683 | +0.028 |
| + 4-class classifier, alpha 0.4 | 0.686 | +0.003 |
| **+ 8-class classifier, alpha 0.4** | **0.693** | **+0.010** |

Every stage was kept only because it raised the validation score.

### 3.3 Where the score is lost: wrong class, not missed vehicles

Ignoring the class, the fused boxes score 0.92 AP50; with classes, 0.69. Most of the loss is look-alike
classes (confusion at score ≥ 0.3):

| True class | AP50 | Main mistake |
|---|---|---|
| Car | 0.94 | – |
| Motorcycle | 0.86 | missed tiny objects |
| Tuktuk | 0.87 | – |
| Bus | 0.76 | 13 of 67 called Truck |
| Van | 0.75 | called Bus / Truck / Car |
| Truck | 0.69 | called Car (29), Songthaew (27), Bus (14) |
| Pickup | 0.50 | 61 of 114 called Car |
| Songthaew | 0.17 | 10 of 24 called Truck; 91 of its 117 train boxes come from one camera (1426) |

### 3.4 The classifier: a second opinion

The first classifier knew only 4 classes (Motorcycle, Tuktuk, Pickup, Songthaew), so it could never say
"this Car is really a Pickup". It was retrained on all 8 classes and now re-checks boxes labelled Car,
Truck, Bus, Pickup, Songthaew and Van (Motorcycle and Tuktuk are already accurate and are left alone).

It never replaces the detector's label. For each checked box:

score(class) = detector score × (0.6 × [class is the detector's label] + 0.4 × classifier probability)

and a class is written as an extra box only if its weight is ≥ 0.05, i.e. the classifier gives it ≥ 12.5%.
Example: detector "Car 0.80", classifier "70% Pickup, 30% Car" → Car 0.58 and Pickup 0.22. A right second
guess rescues a rare class; a wrong one sits low in the ranking and costs almost nothing.

Alpha 0.4 was chosen on validation (0.2: 0.689, 0.4: 0.693, 0.6: 0.682).

### 3.5 Choosing epochs and model size for Round 2

Round 2 trains on all 15 cameras, so there is no clean validation set and the **last epoch** is what gets
used. Shorter runs were tested to find settings whose last epoch is close to their best:

| Run | Best val mAP50 (epoch) | Last epoch |
|---|---|---|
| RF-DETR Large, 40 epochs | 0.635 (6) | 0.577 |
| **RF-DETR Large, 12 epochs** | 0.629 (11) | **0.629** |
| RF-DETR Medium, 12 epochs | 0.628 (6) | 0.608 |
| YOLO26l, 25 epochs | 0.624 (13) | 0.591 |
| **YOLO26m, 25 epochs** | 0.628 (20) | **0.613** |
| YOLO26s, 25 epochs | 0.615 (17) | 0.597 |

Chosen: **RF-DETR Large × 12 epochs** and **YOLO26m × 25 epochs**. Using only their last epochs, the full
pipeline scores **0.688** on validation, against 0.693 for the Round 1 models at their hand-picked best epochs.

## 4. Kaggle submission: the 0.00000 score

The first two uploads scored exactly 0.00000 although the file followed the spec. Ruled out one by one:
encoding (byte-identical to the sample, no BOM), columns, x1/y1/x2/y2 format, unique ids, row order, file
size (a 27k-row file also scored 0), class ids, and the data version (Kaggle's current data was downloaded
and all 4,006 files compared: identical).

**Cause:** `image_id`. `sample_submission.csv` lists long Thai names
(`dataset_1068_annotated_coco1.0_1068_<Thai>_20260825_060110.jpg`), but Kaggle's answer key uses the short
file names, the same style as `train.csv` (`1068_20260825_060110.jpg`). The same predictions with short
names scored **0.63734**. `make_submission.py` now always writes the short names.

The gap from validation (0.693) to public (0.637) is expected: the test cameras are new, and the public
leaderboard uses only about 50% of the test images.

## 5. Round 2: final models (all 15 cameras)

Trained 2026-10-04, about 22 minutes in total (RF-DETR 10 min, YOLO26m 8 min, classifier 3 min,
inference 1 min). Settings, all chosen in Round 1:

| Step | Setting |
|---|---|
| Data | `prepare_data.py --full`, same oversampling |
| RF-DETR Large | 12 epochs, last-epoch EMA weights (`last_ema.pth`) |
| YOLO26m | 25 epochs, early stopping off, last-epoch weights (`last.pt`) |
| Classifier | 8 classes, last epoch |
| Inference | flip TTA, WBF 2:1, classifier alpha 0.4 on Car/Truck/Bus/Pickup/Songthaew/Van |
| Output | `submissions/wbf_final_c8big_a0.4.csv` |

The file has 99,469 boxes for all 997 scored images (at most 100 per image). Its class mix at score ≥ 0.3
is close to the Round 1 file's (for example 5,799 vs 5,711 Car and 108 vs 116 Pickup), so nothing broke.
There is no local score: the validation cameras are now in training.

| Submission | Models | Val mAP50 | Kaggle public |
|---|---|---|---|
| `wbf_exp_c8big_a0.4.csv` | Round 1 (12 cameras) | 0.693 | 0.637 |
| `wbf_final_c8big_a0.4.csv` | Round 2 (15 cameras) | – | **0.703** (1st place) |

Training on all 15 cameras added 0.066 on the public leaderboard. More camera variety is what the unseen
test cameras need most.

## 6. Next steps

- **External data for rare classes:** about 10 Roboflow datasets with Tuktuk, Songthaew and Bus images were
  collected. They will be tested on the Round 1 split first, so their effect can be measured on validation.
  Main risks: other datasets name classes differently, and single-class datasets leave the other vehicles
  in their images unlabelled, which would teach the detector that those vehicles are background.
- **Exam report:** the Kaggle screenshot (Chapter 4) and the error analysis in 3.3 (Chapter 5).
