"""Score validation predictions with pycocotools mAP@50 (the Kaggle metric) and compare pipeline stages.

    python src/evaluate.py --preds y26m_640
    python src/evaluate.py --preds rfdm_640 y26m_640 wbf_exp wbf_exp_cls_a0.5    # ablation table

Reads preds/<run>/val.csv (made by predict_detector.py, fuse.py or reclassify.py). Prints one row per run:
mAP50 (8-class mean, the Kaggle number), agnostic_AP50 (class ignored = box quality only) and per-class
AP50. The confusion matrix of the last run shows which classes get mixed up (score >= --cm-score).
"""
import argparse

import numpy as np
import pandas as pd

from common import CLASS_NAMES, PREDS_DIR, box_iou, build_val_coco, cap_per_image, coco_ap50, load_gt, read_preds, \
    val_images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", nargs="+", required=True, help="prediction folders under preds/")
    p.add_argument("--max-det", type=int, default=100, help="boxes kept per image, as in the submission")
    p.add_argument("--cm-score", type=float, default=0.3, help="score threshold for the confusion matrix")
    return p.parse_args()


def confusion(preds, gt, score_thr, iou_thr=0.5):
    """Rows = true class (+ 'background' for false boxes), cols = predicted class (+ 'missed')."""
    n = len(CLASS_NAMES)
    cm = np.zeros((n + 1, n + 1), dtype=int)
    preds = preds[preds.score >= score_thr]
    p_by, g_by = dict(list(preds.groupby("image"))), dict(list(gt.groupby("image_id")))
    for img in set(p_by) | set(g_by):
        p = p_by.get(img, preds.iloc[:0]).sort_values("score", ascending=False)
        g = g_by.get(img, gt.iloc[:0])
        pb, gb = p[["x1", "y1", "x2", "y2"]].to_numpy(float), g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        pc, gc = p.class_id.to_numpy(), g.class_id.to_numpy()
        iou = box_iou(pb, gb) if len(pb) and len(gb) else np.zeros((len(pb), len(gb)))
        used = np.zeros(len(gb), bool)
        for i in range(len(pb)):  # greedy, highest score first, class ignored when matching
            cand = np.where(~used & (iou[i] >= iou_thr))[0]
            if len(cand):
                j = cand[iou[i, cand].argmax()]
                used[j] = True
                cm[gc[j], pc[i]] += 1
            else:
                cm[n, pc[i]] += 1
        for j in np.where(~used)[0]:
            cm[gc[j], n] += 1
    return pd.DataFrame(cm, index=CLASS_NAMES + ["background"], columns=CLASS_NAMES + ["missed"])


def main():
    args = parse_args()
    images = val_images()
    coco, id_of = build_val_coco(images)
    gt = load_gt()
    gt = gt[gt.image_id.isin(set(images))]

    rows = {}
    for run in args.preds:
        preds = cap_per_image(read_preds(run, "val"), args.max_det)
        m, per_class = coco_ap50(coco, id_of, preds)
        agnostic, _ = coco_ap50(coco, id_of, preds, use_cats=False)
        rows[run] = {"mAP50": m, "agnostic_AP50": agnostic, **dict(zip(CLASS_NAMES, per_class))}
        pd.Series(rows[run]).to_csv(PREDS_DIR / run / "val_ap50.csv", header=["AP50"])

    n_gt = gt.class_id.value_counts()
    print(f"Val: {len(images)} images | GT boxes: "
          + ", ".join(f"{c}={n_gt.get(i, 0)}" for i, c in enumerate(CLASS_NAMES)))
    print(pd.DataFrame(rows).T.round(4).to_string())
    print(f"\nConfusion matrix of {args.preds[-1]} (score >= {args.cm_score}; rows = true, cols = predicted):")
    print(confusion(cap_per_image(read_preds(args.preds[-1], "val"), args.max_det), gt, args.cm_score).to_string())


if __name__ == "__main__":
    main()
