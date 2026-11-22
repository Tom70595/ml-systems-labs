"""Task 5: compare {SGD+momentum, Adam} x {StepLR, Cosine, ReduceLROnPlateau}, 10 epochs."""
import argparse
import csv
import itertools
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


def build_opt_sched(cfg, model, opt_name, sched_name, epochs):
    if opt_name == "SGD":
        opt = optim.SGD(model.parameters(), lr=cfg["training"]["lr"], momentum=0.9)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    if sched_name == "StepLR":
        sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    elif sched_name == "Cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif sched_name == "ReduceLROnPlateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    else:
        sched = None
    return opt, sched


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def run(cfg, opt_name, sched_name, epochs, subset=None):
    set_seed(cfg["seed"])
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(cfg, subset=subset)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    opt, sched = build_opt_sched(cfg, model, opt_name, sched_name, epochs)
    curve = []
    for ep in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
        acc = evaluate(model, val_loader, device)
        if sched is not None:
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(acc)
            else:
                sched.step()
        curve.append(acc)
        print(f"  {opt_name}/{sched_name} epoch {ep+1}/{epochs}: val_acc {acc:.4f}")
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="lab06/configs/base.yaml")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--subset", type=int, default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows = []
    plt.figure(figsize=(10, 6))
    for opt_name, sched_name in itertools.product(["SGD", "Adam"], ["StepLR", "Cosine", "ReduceLROnPlateau"]):
        print(f"\n=== {opt_name} / {sched_name} ===")
        curve = run(cfg, opt_name, sched_name, args.epochs, subset=args.subset)
        for ep, acc in enumerate(curve, 1):
            rows.append({"optimizer": opt_name, "scheduler": sched_name, "epoch": ep, "val_acc": acc})
        plt.plot(range(1, len(curve) + 1), curve, marker="o", label=f"{opt_name}/{sched_name}")

    plt.xlabel("epoch")
    plt.ylabel("val accuracy")
    plt.title("Task 5: optimizer & scheduler comparison")
    plt.legend()
    plt.grid(True)
    os.makedirs("lab06/results", exist_ok=True)
    plt.savefig("lab06/results/optim.png", dpi=120, bbox_inches="tight")

    with open("lab06/results/optim.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["optimizer", "scheduler", "epoch", "val_acc"])
        w.writeheader()
        w.writerows(rows)
    print("\nSaved lab06/results/optim.csv, optim.png")


if __name__ == "__main__":
    main()