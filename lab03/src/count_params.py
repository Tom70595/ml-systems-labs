import torch
import sys
sys.path.append('lab03/src')
from models import MLP, CNN, AttentionModel
from torchinfo import summary

dummy_input = torch.randn(1, 1, 28, 28)

for name, model in [("MLP", MLP()), ("CNN", CNN()), ("Attention", AttentionModel())]:
    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")
    summary(model, input_size=(1, 1, 28, 28))