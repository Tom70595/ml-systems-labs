"""Task 3: vary num_workers [0,1,2,4,8]; find bottleneck; plot throughput."""
import argparse
import csv
import os
import sys
import time

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


def run_one(cfg, num_workers, epochs=2):
    cfg = {**cfg, "data": {**cfg["data"], "num_workers": num_workers}}
    set_seed(cfg["seed"])
    device = get_device()
    train_loader, _, _ = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
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
    dt = time.perf_counter() - t0
    return n * epochs / dt  # samples/sec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="lab06/configs/base.yaml")
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    workers = [0, 1, 2, 4, 8]
    rows = []
    for nw in workers:
        thr = run_one(cfg, nw, epochs=args.epochs)
        print(f"num_workers={nw}: {thr:.1f} samples/s")
        rows.append({"num_workers": nw, "throughput": thr})

    os.makedirs("lab06/results", exist_ok=True)
    with open("lab06/results/workers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["num_workers", "throughput"])
        w.writeheader()
        w.writerows(rows)

    xs = [r["num_workers"] for r in rows]
    ys = [r["throughput"] for r in rows]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("num_workers")
    plt.ylabel("throughput (samples/s)")
    plt.title("Task 3: DataLoader worker count vs throughput")
    plt.grid(True)
    plt.savefig("lab06/results/workers.png", dpi=120, bbox_inches="tight")
    best = max(rows, key=lambda r: r["throughput"])
    print(f"\nBest: num_workers={best['num_workers']} @ {best['throughput']:.1f} samples/s")
    print("Saved lab06/results/workers.csv, workers.png")


if __name__ == "__main__":
    main()