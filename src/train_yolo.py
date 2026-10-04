"""Train the YOLO26 detector (detector 2).

    python src/train_yolo.py --name y26m_640                                    # experiment (camera split)
    python src/train_yolo.py --data datasets/yolo_full/data.yaml --name y26m_640_full   # final model

Weights end up in runs/<name>/weights/best.pt (and last.pt). All arguments are
saved by Ultralytics in runs/<name>/args.yaml for reproducibility.
"""
import argparse

from ultralytics import YOLO

from common import ROOT, RUNS_DIR, YOLO_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26m.pt")
    p.add_argument("--data", default=str(YOLO_DIR / "data.yaml"))
    p.add_argument("--name", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
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
