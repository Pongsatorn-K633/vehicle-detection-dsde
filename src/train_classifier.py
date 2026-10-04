"""Train the second-stage crop classifier (ConvNeXt-Tiny) for the easily confused classes.

    python src/train_classifier.py --name cls_convnext            # experiment (val cameras held out)
    python src/train_classifier.py --name cls_convnext_full --full # final model (all cameras)

Crops come straight from train.csv ground truth. During training each box is randomly shifted and
resized a little, so the classifier also learns from boxes that are slightly off, like detector output.
Rare classes are sampled more often (weight = count^-balance). Saves runs/<name>/best.pt
(best val epoch; with --full the last epoch, because the val cameras are then part of training).
"""
import argparse
import random

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from tqdm import tqdm

from common import CLASS_ID, RUNS_DIR, TRAIN_IMG_DIR, VAL_CAMERAS, camera_of, load_gt

DEFAULT_CLASSES = ["Motorcycle", "Tuktuk", "Pickup", "Songthaew"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--full", action="store_true", help="also train on the val cameras (final model)")
    p.add_argument("--model", default="convnext_tiny")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--pad", type=float, default=0.15, help="context added around each box (fraction of w/h)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--balance", type=float, default=0.5, help="0 = natural frequencies, 1 = fully class-balanced")
    p.add_argument("--samples-per-epoch", type=int, default=6000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def crop_box(img: Image.Image, box, pad: float) -> Image.Image:
    """Crop xyxy box plus `pad` context on every side, clipped to the image."""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    W, H = img.size
    l, t = max(0, int(np.floor(x1 - pw))), max(0, int(np.floor(y1 - ph)))
    r, b = min(W, int(np.ceil(x2 + pw))), min(H, int(np.ceil(y2 + ph)))
    return img.crop((l, t, max(r, l + 1), max(b, t + 1)))


def eval_transform(img_size):
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def jitter(box, rng):
    """Randomly move the box centre by up to 10% and resize each side by 85-120%."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2 + rng.uniform(-0.1, 0.1) * w, (y1 + y2) / 2 + rng.uniform(-0.1, 0.1) * h
    w, h = w * rng.uniform(0.85, 1.2), h * rng.uniform(0.85, 1.2)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class CropDataset(Dataset):
    def __init__(self, df, img_size, pad, train):
        self.items = list(zip(df.image_id, df[["x1", "y1", "x2", "y2"]].to_numpy(float), df.label))
        self.pad, self.train = pad, train
        self.tf = T.Compose([
            T.Resize((img_size, img_size)), T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.3, 0.02),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ]) if train else eval_transform(img_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, box, label = self.items[i]
        img = Image.open(TRAIN_IMG_DIR / fname).convert("RGB")
        if self.train:
            box = jitter(box, random)
        return self.tf(crop_box(img, box, self.pad)), label


def evaluate(model, loader, n_cls, device):
    model.eval()
    conf = np.zeros((n_cls, n_cls), dtype=int)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            np.add.at(conf, (y.numpy(), pred), 1)
    recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
    return recall.mean(), recall, conf


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RUNS_DIR / args.name
    out.mkdir(parents=True, exist_ok=False)

    df = load_gt()
    df = df[df.class_id.isin([CLASS_ID[c] for c in args.classes])].copy()
    df["label"] = df.class_id.map({CLASS_ID[c]: i for i, c in enumerate(args.classes)})
    in_val = df.image_id.map(camera_of).isin(VAL_CAMERAS)
    train_df, val_df = (df if args.full else df[~in_val]), df[in_val]
    counts = train_df.label.value_counts().sort_index()
    print(pd.DataFrame({"train": counts, "val": val_df.label.value_counts().sort_index()})
          .set_axis(args.classes).to_string())

    weights = train_df.label.map(counts.astype(float) ** -args.balance).to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(weights), args.samples_per_epoch, replacement=True)
    train_dl = DataLoader(CropDataset(train_df, args.img_size, args.pad, True), batch_size=args.batch,
                          sampler=sampler, num_workers=args.workers, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(CropDataset(val_df, args.img_size, args.pad, False), batch_size=args.batch * 2,
                        num_workers=args.workers)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(args.classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.1)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_dl, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        bal_acc, recall, conf = evaluate(model, val_dl, len(args.classes), device)
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.3f}  val balanced acc {bal_acc:.3f}  "
              + "  ".join(f"{c}={r:.2f}" for c, r in zip(args.classes, recall)))
        if bal_acc > best or args.full:  # full mode has no clean val, so it keeps the last epoch
            best = bal_acc
            torch.save({"model": args.model, "classes": args.classes, "img_size": args.img_size, "pad": args.pad,
                        "state_dict": model.state_dict()}, out / "best.pt")
            best_conf = conf
    print(f"\n{'Last' if args.full else 'Best'} val balanced accuracy {best:.3f}"
          f"{' (leaky: val cameras were trained on)' if args.full else ''}. Confusion (rows = true, cols = predicted):")
    print(pd.DataFrame(best_conf, index=args.classes, columns=args.classes).to_string())
    print(f"Saved {out / 'best.pt'}")


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
