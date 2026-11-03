import torch
import torch.nn as nn
import time

print("--- Task 6: Eager vs Compiled ---")
# A simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 100)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(64, 100) # Batch size 64, 100 features

# 1. Eager Execution
start = time.time()
for _ in range(1000):
    y = model(x)
eager_time = time.time() - start
print(f"Eager Execution Time: {eager_time:.4f} s")

# 2. Compiled/Traced Execution
# Note: torch.compile takes a long time on the first run (warm-up)
try:
    # We use torch.jit.trace for Codespaces since torch.compile requires a C++ compiler setup
    traced_model = torch.jit.trace(model, x)
    
    # Warm-up
    for _ in range(10): traced_model(x) 
    
    start = time.time()
    for _ in range(1000):
        y = traced_model(x)
    compiled_time = time.time() - start
    print(f"Traced Execution Time: {compiled_time:.4f} s")
    print(f"Speedup: {eager_time / compiled_time:.2f}x faster")
except Exception as e:
    print(f"Compilation failed: {e}")