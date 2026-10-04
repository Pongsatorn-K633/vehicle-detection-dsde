"""Train the YOLO26 detector (detector 2).

    python src/train_yolo.py --name y26l_704                                    # experiment (camera split)
    python src/train_yolo.py --data datasets/yolo_full/data.yaml --name y26l_704_full   # final model

Defaults are for a 32 GB GPU (RTX 5090). On an 8 GB GPU add: --model yolo26m.pt --batch 8 --workers 4

Weights end up in runs/<name>/weights/best.pt (and last.pt). All arguments are
saved by Ultralytics in runs/<name>/args.yaml for reproducibility.

Settings chosen for this dataset (everything else is the Ultralytics default):
- YOLO26l (COCO-pretrained): the large model; more accurate than m and affordable on a 32 GB GPU.
- imgsz 704: images are enlarged 2x because objects are tiny (same as RF-DETR).
- scale 0.3 (default 0.5): random zoom-out is milder, so 7 px motorcycles are not shrunk to 3 px.
- mosaic on, switched off for the last 10 epochs; horizontal flip only (CCTV is always upright).
- cosine LR, 60 epochs with early stop after 20 epochs without improvement.
- batch 16 (~14 GB estimated for yolo26l; 32 would come close to 32 GB). Ultralytics accumulates gradients
  to an effective batch of 64 whatever the batch size, so a smaller GPU (yolo26m: 5.8 GB at batch 8)
  trains the same way.
"""
import argparse

from ultralytics import YOLO

from common import ROOT, RUNS_DIR, YOLO_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26l.pt")
    p.add_argument("--data", default=str(YOLO_DIR / "data.yaml"))
    p.add_argument("--name", required=True)
    p.add_argument("--imgsz", type=int, default=704)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--scale", type=float, default=0.3, help="random resize gain range +-scale")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0, help="use part of the train set (smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(ROOT / args.data),  # absolute paths pass through unchanged
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        fraction=args.fraction,
        scale=args.scale,
        fliplr=0.5,
        flipud=0.0,
        deterministic=True,
        cos_lr=True,
        close_mosaic=10,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
