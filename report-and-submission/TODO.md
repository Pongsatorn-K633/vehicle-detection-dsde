# Still to do

Status on 2026-10-06: final result **0.707** public (`wbf_final_ext_a0.4.csv`), 1st place. Report drafted
(`report.docx`, `report.pdf`). This list replaces `docs/TODO.md`.

| Deadline | What |
|---|---|
| **Sat 10 Oct 2026** | Kaggle closes |
| **Sat 17 Oct 2026** | package due on MCV; GitHub and Google Drive must not change afterwards |

## 1. Kaggle (before 10 Oct, 2 minutes)

- [ ] Submissions tab → tick both final submissions (currently only the 0.703 one is ticked):
  - `wbf_final_ext_a0.4.csv` (public 0.707)
  - `wbf_final_c8big_a0.4.csv` (public 0.703)
- [ ] Take a screenshot of the **final leaderboard** for the report (step 4).
- No more tuning on the public score: only about 50% of the test images are public.

## 2. GitHub

The TA opens the repository's default branch (`main`). All external-data work is on the branch `external-data`.

- [ ] Commit the `report-and-submission/` folder.
- [ ] Push the branch: `git push origin external-data`.
- [ ] Merge `external-data` into `main` and push `main`, so the TA sees the final code and README.
- [ ] Check https://github.com/Pongsatorn-K633/vehicle-detection-dsde opens while logged out (public, or the TA added).

## 3. Google Drive (weights and prepared data, too large for git)

- [ ] Zip and upload the final weights (277 MB):
  - `runs/rfdl_e12_full/last_ema.pth` (RF-DETR Large)
  - `runs/y26m_e25_full/weights/last.pt` (YOLO26m)
  - `runs/cls_all8_ext_full/best.pt` (ConvNeXt-Tiny, with external data)
- [ ] Zip and upload the prepared data: `datasets/yolo_full`, `datasets/rfdetr_full`, `datasets/external`.
- [ ] Share both as "anyone with the link can view".
- [ ] Paste the links into the repository README, with where to put the files to run prediction without training.

## 4. Report (`report.docx`)

- [ ] Chapter 4.3: replace the red placeholder under Figure 4.1 with the final leaderboard screenshot.
- [ ] Appendix, Table A.1: paste the two Google Drive links.
- [ ] Section 3.5: check the course policy on AI coding assistants; add a sentence if it has to be declared.
- [ ] Read it through and edit anything you want in your own words.
- [ ] Re-export the PDF from Word (File → Save As → PDF) and replace `report.pdf`.

## 5. Reproducibility check (recommended, about 30 minutes)

The PDF requires the result to be reproducible and "similar (close) to the result on Kaggle".

- [ ] Fresh clone into a new folder; follow README section 4 (Round 2) and "External data for the classifier".
- [ ] Compare the new CSV with `wbf_final_ext_a0.4.csv`: about 99.5k rows, similar boxes per class at score ≥ 0.3.
- GPU training is not bit-exact, so small differences are normal.

## 6. Submit on MCV (before 17 Oct)

- [ ] `report.docx` and `report.pdf`
- [ ] the uploaded CSV(s) from this folder
- [ ] the GitHub link and the two Google Drive links
- [ ] After submitting, do not push to GitHub or edit the Drive files.
