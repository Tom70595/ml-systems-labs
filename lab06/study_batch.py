"""Task 4: batch-size sweep [32,64,128,256,512]: throughput, peak mem, 5-epoch val acc."""
import argparse
import csv
import os
import sys
import time

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from data import get_dataloaders
from utils import set_seed, get_device


def run_one(cfg, batch_size, epochs, subset=None):
    cfg = {**cfg, "data": {**cfg["data"], "batch_size": batch_size}}
    set_seed(cfg["seed"])
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(cfg, subset=subset)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    proc = psutil.Process(os.getpid())
    peak_mem = 0
    n = len(train_loader.dataset)
    model.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            peak_mem = max(peak_mem, proc.memory_info().rss)
    dt = time.perf_counter() - t0
    throughput = n * epochs / dt

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    return throughput, peak_mem / 1e6, val_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="lab06/configs/base.yaml")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--subset", type=int, default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows = []
    for bs in [32, 64, 128, 256, 512]:
        thr, mem, acc = run_one(cfg, bs, args.epochs, subset=args.subset)
        print(f"batch={bs}: {thr:.1f} samples/s | peak {mem:.0f} MB | val_acc {acc:.4f}")
        rows.append({"batch_size": bs, "throughput": thr, "peak_mem_mb": mem, "val_acc": acc})

    os.makedirs("lab06/results", exist_ok=True)
    with open("lab06/results/batch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch_size", "throughput", "peak_mem_mb", "val_acc"])
        w.writeheader()
        w.writerows(rows)

    xs = [r["batch_size"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xs, [r["throughput"] for r in rows], marker="o"); axes[0].set_title("Throughput"); axes[0].set_xlabel("batch_size"); axes[0].set_ylabel("samples/s")
    axes[1].plot(xs, [r["peak_mem_mb"] for r in rows], marker="o"); axes[1].set_title("Peak RSS"); axes[1].set_xlabel("batch_size"); axes[1].set_ylabel("MB")
    axes[2].plot(xs, [r["val_acc"] for r in rows], marker="o"); axes[2].set_title(f"Val acc after {args.epochs} epochs"); axes[2].set_xlabel("batch_size"); axes[2].set_ylabel("accuracy")
    for ax in axes: ax.grid(True)
    plt.tight_layout()
    plt.savefig("lab06/results/batch.png", dpi=120, bbox_inches="tight")
    print("Saved lab06/results/batch.csv, batch.png")


if __name__ == "__main__":
    main()