# Thai vehicle detection — pipeline architecture

Detect and classify vehicles in **still images** into **8 classes**, including Thai vehicles (tuktuk, songthaew). Accuracy is the priority (university midterm).

## Pipeline

```
Input image
  -> Test-time augmentation (original + horizontally flipped)
  -> Detector 1: RF-DETR   |   Detector 2: YOLO26     (both: 8 classes)
  -> Weighted Boxes Fusion (merge all predictions into one set)
  -> For each box:
       label in {pickup, songthaew, motorcycle, tuktuk}?
         yes -> crop (+15% padding) -> 224x224 -> ConvNeXt-Tiny -> final label
         no  -> keep detector label
  -> Final detections
```

**Why each stage**
- **Two detectors:** different architectures (transformer vs CNN) make different mistakes; merging them improves accuracy.
- **TTA:** predicting on the flipped image too gives a small extra boost.
- **WBF:** averages overlapping boxes from all predictions, weighted by confidence. Boxes both models agree on score highest.
- **Second-stage classifier:** songthaew looks like a pickup and tuktuk looks like a motorcycle. A classifier focused only on these 4 classes fixes the detector's label mistakes.

## Models to train

| Model | Role | Package | Suggested size | Trained on |
|---|---|---|---|---|
| RF-DETR | Detector 1 (main, most accurate) | `rfdetr` | Medium or Large | All 8 classes |
| YOLO26 | Detector 2 | `ultralytics` | `yolo26m` or `yolo26l` | All 8 classes |
| ConvNeXt-Tiny | Classifier for confusable classes | `timm` (`convnext_tiny`, pretrained) | Tiny | GT crops of pickup, songthaew, motorcycle, tuktuk |

Alternative classifiers: EfficientNetV2-S (lighter), or frozen DINOv2 + linear layer (good when rare classes have few images).

## Suggestions

**Training**
- Class IDs must be identical across both detectors (check RF-DETR outputs; some exports shift IDs by 1).
- Rare classes (tuktuk, songthaew): oversample, augment, collect more images if possible.
- Add hard negatives: pickups with canopies, motorcycle sidecar carts, three-wheel delivery vehicles.
- Classifier crops: ~15% padding for context (roof, rear seats), clip to image bounds, class-balanced sampling.
- Split train/val so images from the same camera/location don't appear in both.

**WBF** (`ensemble-boxes` package)
- Predict with low confidence threshold (0.01–0.05); don't filter before fusion.
- Un-flip boxes from the flipped image: `x1' = W - x2`, `x2' = W - x1`.
- Normalize boxes to 0..1 before fusion, convert back after.
- Start with `weights=[2, 2, 1, 1]` (RF-DETR orig/flip, YOLO orig/flip), `iou_thr=0.6`, `skip_box_thr=0.01`; tune on validation.

**Evaluation**
- Report mAP50, mAP50-95, per-class AP, and a confusion matrix.
- Ablation on validation: RF-DETR alone → YOLO26 alone → WBF → WBF + TTA → WBF + TTA + classifier. Keep a stage only if it helps.

**Short on time?** Train RF-DETR well with good data first. It gives most of the score; the rest are extra points.
