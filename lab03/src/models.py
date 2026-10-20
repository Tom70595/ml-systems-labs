import torch
import torch.nn as nn

# ==========================================
# 1. MLP (Multi-Layer Perceptron)
# Adjusted to ~135k params
# ==========================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 128),    # 100,480
            nn.ReLU(),
            nn.Linear(128, 256),    # 33,024
            nn.ReLU(),
            nn.Linear(256, 10)      # 2,570
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


# ==========================================
# 2. CNN (Convolutional Neural Network)
# Adjusted to ~135k params (added a wider linear layer)
# ==========================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 80),  # 125,520
            nn.ReLU(),
            nn.Linear(80, 10)           # 810
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 3. Attention (Vision Transformer-style)
# Adjusted to ~135k params (wider d_model)
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=7, embed_dim=80):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=80, nhead=4, dim_feedforward=160, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(80, 10)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)