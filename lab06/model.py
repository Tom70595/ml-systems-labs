"""CNN copied from lab03/src/models.py — best model from Lab 3 (89.47% test acc)."""
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 80),
            nn.ReLU(),
            nn.Linear(80, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg) -> nn.Module:
    name = cfg["model"]["name"]
    num_classes = cfg["model"].get("num_classes", 10)
    if name == "CNN":
        return CNN(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")