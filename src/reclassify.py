"""Second stage: re-score the labels of easily confused boxes with the crop classifier.

    python src/reclassify.py --preds wbf_exp --classifier runs/cls_convnext/best.pt --alpha 0 0.3 0.5 0.7
    python src/reclassify.py --preds wbf_final --classifier runs/cls_convnext_full/best.pt --alpha 0.5 --splits test

For every box whose label is in --apply-to (default: the classifier's classes) the crop is classified
and the box is written once per candidate class c with

    score_c = detector_score * (alpha * p_classifier(c) + (1 - alpha) * [c == detector label])

so instead of overwriting the label, the classifier adds lower-scored "second guess" boxes. mAP ranks
boxes by score, so a wrong second guess costs little while a right one can rescue a rare class.
alpha = 0 reproduces the input exactly; each alpha is written to preds/<out>_a<alpha>/<split>.csv.
Other boxes, and boxes scoring below --min-score (they barely affect AP), are copied unchanged.
"""
import argparse

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision.ops import batched_nms
from tqdm import tqdm

from common import CLASS_ID, PRED_COLUMNS, image_dir_of, read_preds, write_preds
from train_classifier import crop_box, eval_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="prediction folder under preds/ (usually the WBF output)")
    p.add_argument("--classifier", required=True)
    p.add_argument("--alpha", type=float, nargs="+", default=[0.5], help="trust in the classifier, 0..1")
    p.add_argument("--apply-to", nargs="+", default=None, help="detector labels to re-check (default: all "
                                                               "classifier classes)")
    p.add_argument("--min-score", type=float, default=0.05, help="only re-check boxes at least this confident")
    p.add_argument("--min-frac", type=float, default=0.05, help="skip candidate classes whose blended weight "
                                                                "is below this")
    p.add_argument("--nms-iou", type=float, default=0.7, help="merge same-class duplicates created by the stage")
    p.add_argument("--out", default=None, help="default <preds>_cls")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()


def load_classifier(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def classify(model, ckpt, df, device, batch):
    """Classifier probabilities [len(df), n_classes] for the boxes in df (sorted by image)."""
    tf = eval_transform(ckpt["img_size"])
    probs = np.zeros((len(df), len(ckpt["classes"])), dtype=np.float32)
    crops, img, img_name = [], None, None
    for i, r in enumerate(tqdm(df.itertuples(), total=len(df), desc="classify", leave=False)):
        if r.image != img_name:  # rows are sorted by image, so each image is opened once
            img_name = r.image
            img = Image.open(image_dir_of(img_name) / img_name).convert("RGB")
        crops.append(tf(crop_box(img, (r.x1, r.y1, r.x2, r.y2), ckpt["pad"])))
        if len(crops) == batch or i == len(df) - 1:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                p = model(torch.stack(crops).to(device)).float().softmax(1)
            probs[i + 1 - len(crops):i + 1] = p.cpu().numpy()
            crops = []
    return probs


def rescore(df, probs, cls_ids, alpha, min_frac, nms_iou):
    """Expand each checked box into one box per plausible class, then drop same-class duplicates."""
    det_onehot = (df.class_id.to_numpy()[:, None] == np.array(cls_ids)[None, :]).astype(np.float32)
    blended = alpha * probs + (1 - alpha) * det_onehot
    rows, cols = np.nonzero(blended >= min_frac)
    out = df.iloc[rows][["image", "x1", "y1", "x2", "y2"]].copy()
    out["class_id"] = np.array(cls_ids)[cols]
    out["score"] = df.score.to_numpy()[rows] * blended[rows, cols]
    if out.empty or alpha == 0:  # alpha 0 = detector labels only, nothing new to merge
        return out
    boxes = torch.as_tensor(out[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32))
    groups = torch.as_tensor(pd.factorize(out.image + "_" + out.class_id.astype(str))[0])
    keep = batched_nms(boxes, torch.as_tensor(out.score.to_numpy(dtype=np.float32)), groups, nms_iou)
    return out.iloc[keep.numpy()]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_classifier(args.classifier, device)
    cls_ids = [CLASS_ID[c] for c in ckpt["classes"]]
    apply_ids = [CLASS_ID[c] for c in (args.apply_to or ckpt["classes"])]
    out_base = args.out or f"{args.preds}_cls"

    for split in args.splits:
        df = read_preds(args.preds, split)
        checked = df.class_id.isin(apply_ids) & (df.score >= args.min_score)
        sub = df[checked].sort_values("image", kind="stable").reset_index(drop=True)
        probs = classify(model, ckpt, sub, device, args.batch)
        for alpha in args.alpha:
            new = rescore(sub, probs, cls_ids, alpha, args.min_frac, args.nms_iou)
            res = pd.concat([df[~checked], new[PRED_COLUMNS]], ignore_index=True)
            out = write_preds(res, f"{out_base}_a{alpha:g}", split)
            print(f"{split} alpha={alpha:g}: {checked.sum()} boxes checked, {len(res)} boxes out -> {out}")


if __name__ == "__main__":
    main()
