from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam

from .utils import set_seed, accuracy
from .data import create_loader
from .model import SimpleAudioCNN
from .config import Config

def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)  # [B,1,n_mels,time]
        y = y.to(device)

        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y.detach()) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

def main():
    cfg = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=cfg.train_csv)
    parser.add_argument("--train_wav_dir", default=cfg.train_wav_dir)
    parser.add_argument("--val_csv", default=cfg.val_csv)
    parser.add_argument("--val_wav_dir", default=cfg.val_wav_dir)

    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--seed", type=int, default=cfg.seed)

    parser.add_argument("--clip_seconds", type=float, default=cfg.clip_seconds)
    parser.add_argument("--sample_rate", type=int, default=cfg.sample_rate)

    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train loader
    train_ds, train_loader = create_loader(
        csv_path=args.train_csv,
        wav_dir=args.train_wav_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        num_workers=0,
        shuffle=True,
    )

    # Val loader (optional if file exists)
    val_loader = None
    if os.path.exists(args.val_csv) and os.path.isdir(args.val_wav_dir):
        _, val_loader = create_loader(
            csv_path=args.val_csv,
            wav_dir=args.val_wav_dir,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            num_workers=0,
            shuffle=False,
        )

    model = SimpleAudioCNN(num_classes=train_ds.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, criterion, device)

        msg = f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f}"

        if val_loader is not None:
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)
            msg += f" | val loss {va_loss:.4f} acc {va_acc:.4f}"

        print(msg)

    ckpt_path = os.path.join(args.save_dir, "model_final.pt")
    torch.save({"model_state": model.state_dict(), "num_classes": train_ds.num_classes}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
