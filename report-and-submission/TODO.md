# Still to do

Status on 2026-10-09: Kaggle done (both finals ticked: 0.707 and 0.703). Run scripts, Drive folder and the
reproducibility check done. This list replaces `docs/TODO.md`.

| Deadline | What |
|---|---|
| **Sat 10 Oct 2026** | Kaggle closes |
| **Sat 17 Oct 2026, 23:59** | package due on MCV; GitHub and Google Drive must not change afterwards |

## 1. Kaggle

- [x] Both final submissions ticked: `wbf_final_ext_a0.4.csv` (0.707), `wbf_final_c8big_a0.4.csv` (0.703).
- [ ] Optional, before it closes: submit `../repro-check/kaggle-check/repro_v1_c8big_a0.4.csv` and
      `repro_v2_ext_a0.4.csv` (retrained from scratch) to get a real score for the reproducibility check.
      Do not change the two ticked selections.
- [ ] After it closes: screenshot the **Private** leaderboard (final score) for report Chapter 4.

## 2. Reproducibility check (done 2026-10-09)

Fresh clone + fresh conda env from the pinned `environment.yml`:

| Check | Result |
|---|---|
| `--predict-only` with the Drive weight zips (both versions) | CSVs byte-identical to the submitted ones, ~1.5 min |
| v1 retrained from scratch (RF-DETR at batch 8 x 2 accumulation) | agreement mAP50 0.976 with the submitted CSV, ~25 min |
| v2 retrained from the raw Roboflow sets | 12,308 images / 18,867 boxes as documented; agreement 0.981, ~10 min |

Agreement = mAP@50 of the new CSV scored against the submitted CSV's boxes with confidence >= 0.3. For scale:
v1 vs v2 (same detectors, different classifier) is 0.992; round-1 models vs final models is 0.933.

## 3. Google Drive

Folder to upload: `drive-upload/2110531_DSDE_Midterm_6970180821_Pongsatorn/` (about 2.5 GB, plus the code zip).

- [ ] Create the folder `2110531_DSDE_Midterm_6970180821_Pongsatorn` on Drive, share it as
      "Anyone with the link can view", and copy the link.
- [ ] Put the link into `README.md` (`<GOOGLE_DRIVE_LINK>`) and report Table A.1; commit and push.
- [ ] Build `1_source_code/vehicle-detection-dsde.zip` from the final `main` (after the link is in).
- [ ] Copy the final `report.docx` / `report.pdf` into `5_report/`.
- [ ] Upload everything; open the link in a private browser window to check it works.

## 4. GitHub

- [x] `external-data` merged into `main` (two run scripts, README "Two versions").
- [ ] After the Drive link is committed: `git tag -a v2-external-data main -m "Version 2 (0.707)"` (v1 tag exists).
- [ ] Push: `git push origin main external-data --tags`.
- [ ] Make the repository **public** (Settings → General → Danger Zone → Change visibility); it is private now.
- [ ] Check https://github.com/Pongsatorn-K633/vehicle-detection-dsde opens while logged out.

## 5. Report (`report.docx`)

- [ ] Chapter 4.3: replace the red placeholder under Figure 4.1 with the final (Private) leaderboard screenshot.
- [ ] Appendix, Table A.1: paste the Google Drive link (prepared data and weights rows).
- [ ] Mention the two run scripts and the reproducibility check (section 2 above) if you want.
- [ ] Section 3.5: check the course policy on AI coding assistants; add a sentence if it has to be declared.
- [ ] Read it through, then re-export the PDF from Word and replace `report.pdf`.

## 6. Submit on MCV (before 17 Oct, 23:59)

- [ ] Textbox: the Google Drive link (and the GitHub link).
- [ ] Attach files: `report.pdf` and `report.docx`.
- [ ] Save / Submit. After that, do not push to GitHub or edit the Drive files.
