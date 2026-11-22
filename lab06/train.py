"""YAML-driven training: seeding, checkpoints, MLflow, gradient accumulation, resume.

Task 1: reproducible YAML pipeline + checkpoint saving + MLflow logging.
Task 6: gradient accumulation (--grad-accum N).
Task 7: resume from checkpoint (--resume path).

Examples:
  python lab06/train.py --config lab06/configs/base.yaml
  python lab06/train.py --config lab06/configs/base.yaml --grad-accum 4 --batch 64
  python lab06/train.py --config lab06/configs/base.yaml --resume lab06/checkpoints/epoch_3.pt
"""
import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from data import get_dataloaders
from utils import set_seed, save_checkpoint, load_checkpoint, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="lab06/configs/base.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--subset", type=int, default=None, help="limit train set size (for fast sweeps)")
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--tag", default="", help="suffix for run name")
    return p.parse_args()


def build_optimizer(cfg, model):
    t = cfg["training"]
    if t["optimizer"].lower() == "sgd":
        return optim.SGD(model.parameters(), lr=t["lr"], momentum=t.get("momentum", 0.9))
    return optim.Adam(model.parameters(), lr=t["lr"])


def build_scheduler(cfg, optimizer):
    t = cfg["training"]
    s = t.get("scheduler", "none")
    if s == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=t.get("step_size", 2), gamma=t.get("gamma", 0.5))
    if s == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    if s == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=t.get("gamma", 0.5), patience=1)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum_steps=1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels) / grad_accum_steps
        loss.backward()
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch is not None:
        cfg["data"]["batch_size"] = args.batch
    if args.grad_accum is not None:
        cfg["training"]["grad_accum_steps"] = args.grad_accum
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg, subset=args.subset)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # MLflow
    mlflow = None
    if not args.no_mlflow:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        mlflow.start_run(run_name=f"{cfg['model']['name']}-{args.tag or 'base'}")

    # Log params
    flat_params = {
        "seed": cfg["seed"], "batch_size": cfg["data"]["batch_size"],
        "num_workers": cfg["data"]["num_workers"], "epochs": cfg["training"]["epochs"],
        "lr": cfg["training"]["lr"], "optimizer": cfg["training"]["optimizer"],
        "scheduler": cfg["training"]["scheduler"],
        "grad_accum_steps": cfg["training"]["grad_accum_steps"],
        "model": cfg["model"]["name"],
    }
    if mlflow:
        mlflow.log_params(flat_params)

    os.makedirs("lab06/results", exist_ok=True)
    results_path = f"lab06/results/train_{args.tag or 'base'}.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"])

    epochs = cfg["training"]["epochs"]
    grad_accum = cfg["training"]["grad_accum_steps"]
    ckpt_dir = cfg["checkpoint"]["dir"]

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_accum)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | lr {cur_lr:.6f} | {dt:.1f}s")

        with open(results_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, cur_lr, dt])

        if mlflow:
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc,
                                "val_loss": val_loss, "val_acc": val_acc, "lr": cur_lr,
                                "epoch_time_s": dt}, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % cfg["checkpoint"]["save_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_acc, cfg)
            if mlflow:
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")
    if mlflow:
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        mlflow.log_artifact(results_path, artifact_path="results")
        mlflow.end_run()


if __name__ == "__main__":
    main()