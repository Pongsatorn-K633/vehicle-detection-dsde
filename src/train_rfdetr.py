"""Train the RF-DETR detector (detector 1).

    python src/train_rfdetr.py --name rfdm_640                                   # experiment (camera split)
    python src/train_rfdetr.py --data datasets/rfdetr_full --name rfdm_640_full  # final model

The dataset comes from src/prepare_data.py. Weights end up in runs/<name>/checkpoint_best_total.pth.
Resolution must be a multiple of 64 (images are 352x288, so 640 enlarges them about 1.8x like YOLO).
On an 8 GB GPU, Medium fits with --batch 4 --grad-accum 4 (effective batch 16).
"""
import argparse

import torch
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

from common import CLASS_NAMES, DATASETS_DIR, ROOT, RUNS_DIR

MODELS = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=MODELS, default="medium")
    p.add_argument("--data", default=str(DATASETS_DIR / "rfdetr"))
    p.add_argument("--name", required=True)
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out = RUNS_DIR / args.name
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick another --name")
    model = MODELS[args.size]()
    model.train(
        dataset_dir=str(ROOT / args.data),
        output_dir=str(out),
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.workers,
        seed=args.seed,
        class_names=CLASS_NAMES,
        tensorboard=False,
        progress_bar="tqdm",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Best weights: {out / 'checkpoint_best_total.pth'}")


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
