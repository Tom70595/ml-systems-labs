"""Data pipeline — same normalization/split as lab03/src/data_setup.py."""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


def get_dataloaders(cfg, subset: int | None = None):
    d = cfg["data"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((d["mean"],), (d["std"],)),
    ])

    full_train = datasets.FashionMNIST(
        root=d["root"], train=True, download=True, transform=transform
    )
    test_ds = datasets.FashionMNIST(
        root=d["root"], train=False, download=True, transform=transform
    )

    val_size = d["val_size"]
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    if subset is not None:
        train_ds = Subset(train_ds, range(min(subset, len(train_ds))))
        val_ds = Subset(val_ds, range(min(subset // 5, len(val_ds))))

    nw = d["num_workers"]
    bs = d["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader