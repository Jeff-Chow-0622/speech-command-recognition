from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from .data import create_loader
from .model import SimpleAudioCNN


@torch.no_grad()
def run_eval(model, loader, device: str):
    model.eval()

    all_preds = []
    all_y = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_y.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_y = np.concatenate(all_y)

    acc = (all_preds == all_y).mean()

    # confusion matrix
    num_classes = int(max(all_y.max(), all_preds.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(all_y, all_preds):
        cm[yt, yp] += 1

    return acc, cm


def save_confusion_matrix_png(cm: np.ndarray, out_path: str):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file (e.g., data/scr_val.csv)")
    parser.add_argument("--wav_dir", required=True, help="WAV folder (e.g., data/validation)")
    parser.add_argument("--ckpt", default="checkpoints/model_final.pt", help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_seconds", type=float, default=1.0)

    parser.add_argument("--save_cm", action="store_true", help="Save confusion matrix image to assets/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, loader = create_loader(
        csv_path=args.csv,
        wav_dir=args.wav_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        num_workers=0,
        shuffle=False,
    )

    model = SimpleAudioCNN(num_classes=ds.num_classes).to(device)

    # Load checkpoint if available
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"[WARN] Checkpoint not found: {args.ckpt}. Evaluating with randomly initialized model.")

    acc, cm = run_eval(model, loader, device)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    if args.save_cm:
        out_path = "assets/confusion_matrix.png"
        save_confusion_matrix_png(cm, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
