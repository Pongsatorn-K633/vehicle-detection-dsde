"""Train the RF-DETR detector (detector 1).

    python src/train_rfdetr.py --name rfdl_704                                   # experiment (camera split)
    python src/train_rfdetr.py --data datasets/rfdetr_full --name rfdl_704_full  # final model

Defaults are for a 32 GB GPU (RTX 5090). On an 8 GB GPU add: --batch 4 --grad-accum 4 --workers 2

The dataset comes from src/prepare_data.py. Weights end up in runs/<name>/checkpoint_best_total.pth and the
full resolved settings (incl. RF-DETR defaults) in runs/<name>/training_config.json.

Settings chosen for this dataset (everything else is the RF-DETR default):
- RF-DETR Large: same network as Medium (DINOv2-S backbone, 4 decoder layers, 34M parameters) but
  pretrained at 704 px, the resolution used here, so it costs the same as Medium at 704.
- resolution 704: objects are tiny (median motorcycle 7x17 px), so the 352x288 images are enlarged 2x.
  Must be a multiple of 64.
- cosine LR schedule + 1 warm-up epoch: the default "step" schedule only drops the LR at epoch 100,
  so a 40-epoch run would never decay.
- 40 epochs: the oversampled train set has 33% more images per epoch, so this is ~53 epochs of
  original-size passes.
- augmentation: horizontal flip (default) + brightness/contrast and colour shifts for day/night/rain
  cameras. No vertical flips or rotations: CCTV cameras are always upright.
- batch 16, lr 1e-4 (RF-DETR's reference setting). On 8 GB: batch 4 x 4 accumulation gives the same effective batch.
"""
import argparse

import torch
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

from common import CLASS_NAMES, DATASETS_DIR, ROOT, RUNS_DIR

MODELS = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}

AUGMENT = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
    "HueSaturationValue": {"hue_shift_limit": 5, "sat_shift_limit": 20, "val_shift_limit": 15, "p": 0.3},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=MODELS, default="large")
    p.add_argument("--data", default=str(DATASETS_DIR / "rfdetr"))
    p.add_argument("--name", required=True)
    p.add_argument("--resolution", type=int, default=704)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
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
        lr_encoder=args.lr_encoder,
        lr_scheduler="cosine",
        warmup_epochs=args.warmup_epochs,
        aug_config=AUGMENT,
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
