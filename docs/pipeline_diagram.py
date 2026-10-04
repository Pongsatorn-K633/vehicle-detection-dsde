"""Draw docs/thai_vehicle_detection_pipeline.png (the inference pipeline as it runs for the final submission).

    python docs/pipeline_diagram.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 1640, 2330  # canvas in drawing units (y grows downward)
OUT = Path(__file__).with_name("thai_vehicle_detection_pipeline.png")

STYLES = {  # fill, edge, title colour, subtitle colour
    "plain": ("#FAFAF8", "#D3D1C7", "#1F1F1F", "#5F5E5A"),
    "model": ("#EEEDFE", "#6E62D6", "#3C3489", "#534AB7"),
    "cls": ("#E1F5EE", "#1D7A5F", "#085041", "#0F6E56"),
}
ARROW = "#8A8A86"


def box(ax, cx, cy, w, h, title, sub, style="plain"):
    fill, edge, tcol, scol = STYLES[style]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=22",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(cx, cy - 22, title, ha="center", va="center", fontsize=23, fontweight="semibold", color=tcol)
    ax.text(cx, cy + 26, sub, ha="center", va="center", fontsize=18.5, color=scol)
    return {"top": cy - h / 2, "bottom": cy + h / 2, "x": cx}


def path(ax, pts, arrow=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if arrow else xs, ys[:-1] if arrow else ys, color=ARROW, lw=3, solid_capstyle="butt")
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.45", color=ARROW, lw=3,
                                    shrinkA=0, shrinkB=0))


def label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=18.5, color="#5F5E5A")


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx, left, right = W / 2, 527, 1100
    inp = box(ax, cx, 125, 576, 106,"Input image", "352 × 288 CCTV frame, enlarged to 704 px")
    tta = box(ax, cx, 330, 576, 134, "Test-time augmentation", "Original + mirrored image")
    rf = box(ax, left, 570, 480, 134, "RF-DETR Large", "Detector 1, 8 classes", "model")
    yo = box(ax, right, 570, 480, 134, "YOLO26m", "Detector 2, 8 classes", "model")
    nms = box(ax, cx, 810, 576, 134, "Remove duplicates (NMS)", "Within each model, per class", "model")
    wbf = box(ax, cx, 1050, 576, 134, "Weighted Boxes Fusion", "Merge 4 sets: RF-DETR × 2, YOLO × 1", "model")
    dec = box(ax, cx, 1290, 760, 134, "Confusable class?",
              "Car, Truck, Bus, Pickup, Songthaew, Van (score ≥ 0.05)")
    crop = box(ax, 455, 1570, 528, 134, "Crop + 15% padding", "Resize to 224 × 224", "cls")
    conv = box(ax, 455, 1810, 528, 134, "ConvNeXt-Tiny", "Probabilities for all 8 classes", "cls")
    add = box(ax, 455, 2050, 528, 134, "Add second guess", "Keep detector label; α = 0.4", "cls")
    keep = box(ax, 1220, 1570, 528, 134, "Keep detector label", "Motorcycle, Tuktuk, low scores")
    fin = box(ax, cx, 2250, 576, 106, "Final detections", "≤ 100 boxes per image, 8 classes")

    path(ax, [(cx, inp["bottom"]), (cx, tta["top"] - 4)])
    split = (tta["bottom"] + rf["top"]) / 2
    path(ax, [(cx, tta["bottom"]), (cx, split)], arrow=False)
    path(ax, [(left, split), (right, split)], arrow=False)
    path(ax, [(left, split), (left, rf["top"] - 4)])
    path(ax, [(right, split), (right, yo["top"] - 4)])
    join = (rf["bottom"] + nms["top"]) / 2
    path(ax, [(left, rf["bottom"]), (left, join), (right, join), (right, yo["bottom"])], arrow=False)
    path(ax, [(cx, join), (cx, nms["top"] - 4)])
    path(ax, [(cx, nms["bottom"]), (cx, wbf["top"] - 4)])
    path(ax, [(cx, wbf["bottom"]), (cx, dec["top"] - 4)])

    fork = (dec["bottom"] + crop["top"]) / 2
    path(ax, [(cx, dec["bottom"]), (cx, fork)], arrow=False)
    path(ax, [(crop["x"], fork), (keep["x"], fork)], arrow=False)
    path(ax, [(crop["x"], fork), (crop["x"], crop["top"] - 4)])
    path(ax, [(keep["x"], fork), (keep["x"], keep["top"] - 4)])
    label(ax, crop["x"] - 48, fork + 42, "Yes")
    label(ax, keep["x"] + 48, fork + 42, "No")

    path(ax, [(crop["x"], crop["bottom"]), (crop["x"], conv["top"] - 4)])
    path(ax, [(conv["x"], conv["bottom"]), (conv["x"], add["top"] - 4)])
    path(ax, [(add["x"], add["bottom"]), (add["x"], 2160), (cx - 96, 2160), (cx - 96, fin["top"] - 4)])
    path(ax, [(keep["x"], keep["bottom"]), (keep["x"], 2160), (cx + 96, 2160), (cx + 96, fin["top"] - 4)])

    fig.savefig(OUT, dpi=166, facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
