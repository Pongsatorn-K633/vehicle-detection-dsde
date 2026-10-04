# To do

Status on 2026-10-05: Round 2 submission `submissions/wbf_final_c8big_a0.4.csv` scored **0.703** on the Kaggle
public leaderboard (1st place). Everything up to that point is committed (`47f3a08`), not pushed.

Deadlines: **Kaggle closes Sat 10 Oct 2026.** **Package due Sat 17 Oct 2026** (MCV).

## 1. Kaggle (2 minutes)

- [ ] Submissions tab → select both final submissions:
  - `wbf_final_c8big_a0.4.csv` (public 0.703, Round 2 models)
  - `wbf_exp_c8big_a0.4.csv` (public 0.637, Round 1 models)
- Only selected files count for the final (private) ranking, which uses the other 50% of the test images.
- Limit: 5 submissions per day.

## 2. Package, part 1: code, CSV, prepared data, weights

The TA must be able to open the GitHub / Google Drive links. **Nothing may change after the submission date.**

- [ ] Push the repo to GitHub; check the link works while logged out (public, or TA added).
- [ ] Upload the final weights (not in git, 277 MB) to Google Drive:
  - `runs/rfdl_e12_full/last_ema.pth` (RF-DETR Large)
  - `runs/y26m_e25_full/weights/last.pt` (YOLO26m)
  - `runs/cls_all8_full/best.pt` (ConvNeXt-Tiny classifier)
- [ ] Zip the prepared data (`datasets/yolo_full`, `datasets/rfdetr_full`) to Google Drive.
- [ ] Add the Drive links to the README, plus where to put the files to skip training and run only prediction.
- [ ] The uploaded CSV is already in `submissions/`.

## 3. Reproducibility check (about 25 minutes)

The PDF requires the result to be reproducible and "similar (close) to the result on Kaggle".

- [ ] Fresh clone in a new folder, follow README section 4 (Round 2) exactly.
- [ ] Compare the new CSV with `submissions/wbf_final_c8big_a0.4.csv`: row count (~99.5k), boxes per class at
      score ≥ 0.3, and ideally upload it once to check the score is close to 0.703.
- GPU training is not bit-exact even with seeds, so small differences are normal.

## 4. Package, part 2: report (Word + PDF)

Most content is in `docs/report.md`.

| Chapter | Source |
|---|---|
| 1. Introduction | README dataset table, PDF pages 2–4 |
| 2. Data preparation | report section 1 (camera split, oversampling, augmentation) |
| 3. Model | report section 2 + `docs/thai_vehicle_detection_pipeline.png`; add architecture details for RF-DETR (DINOv2 backbone, DETR decoder), YOLO26m, ConvNeXt-Tiny; pretrained data (COCO, ImageNet); no paid API, so cost = GPU time only |
| 4. Results | report sections 3–5 + **Kaggle leaderboard screenshot** (required) |
| 5. Discussion | report section 3.3 (error analysis: Pickup/Car, Songthaew/Truck), the `image_id` issue, how to improve |
| 6. Conclusion | short |

## 5. External data (optional, decided: skip unless there is spare time)

About 10 Roboflow datasets with Tuktuk / Songthaew / Bus were collected (list in the earlier screenshot).
Not needed for the grade: both Kaggle parts are already at full marks (mAP ≥ 0.56 → 5 points; top 30% → 5 points).

If tried anyway, before 10 Oct:
- [ ] Ask the TA whether external data is allowed.
- [ ] Download in YOLOv8 format into `external-data/<dataset>/` (add `external-data/` to `.gitignore`).
- [ ] Map class names to our 8 ids; fill in unlabelled vehicles with the current models (single-class datasets
      leave cars unlabelled, which would teach the detector they are background).
- [ ] Train on the Round 1 split + external data; keep it only if validation beats **0.688**.
- [ ] Safer variant: add external crops to the classifier only.
- Keep the 0.703 file selected on Kaggle whatever happens.
