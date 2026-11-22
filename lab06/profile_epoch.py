"""Task 2: profile one epoch with torch.profiler; report time share.

Measures (wall-clock) time inside:
  - data loading (next(iter(loader)))
  - forward pass
  - backward pass
  - optimizer step
and also dumps the torch.profiler key_averages table.
"""
import argparse
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
from utils import set_seed, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="lab06/configs/base.yaml")
    p.add_argument("--max-batches", type=int, default=50)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = get_device()

    train_loader, _, _ = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    t_load = t_fwd = t_bwd = t_step = 0.0
    n_batches = 0

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        model.train()
        it = iter(train_loader)
        for i in range(args.max_batches):
            t0 = time.perf_counter()
            try:
                images, labels = next(it)
            except StopIteration:
                it = iter(train_loader)
                images, labels = next(it)
            t1 = time.perf_counter()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            t2 = time.perf_counter()
            loss = criterion(outputs, labels)
            loss.backward()
            t3 = time.perf_counter()
            optimizer.step()
            t4 = time.perf_counter()

            t_load += t1 - t0
            t_fwd += t2 - t1
            t_bwd += t3 - t2
            t_step += t4 - t3
            n_batches += 1

    total = t_load + t_fwd + t_bwd + t_step
    print("\n=== Time breakdown over {} batches ===".format(n_batches))
    for name, t in [("data_loading", t_load), ("forward", t_fwd), ("backward", t_bwd), ("optimizer_step", t_step)]:
        print(f"{name:15s} {t:8.3f} s   {100 * t / total:5.1f}%")
    print(f"{'total':15s} {total:8.3f} s")

    print("\n=== torch.profiler key_averages ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    os.makedirs("lab06/results", exist_ok=True)
    with open("lab06/results/profile.txt", "w") as f:
        f.write(f"batches {n_batches}\n")
        for name, t in [("data_loading", t_load), ("forward", t_fwd), ("backward", t_bwd), ("optimizer_step", t_step)]:
            f.write(f"{name:15s} {t:8.3f} s   {100 * t / total:5.1f}%\n")
        f.write("\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("\nSaved lab06/results/profile.txt")


if __name__ == "__main__":
    main()