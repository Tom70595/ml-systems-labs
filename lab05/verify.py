import torch
from engine import Value

print("--- Task 4: Verifying Gradients ---")
# 1. Our Engine
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = (a * b + c) * 2.0
d = d.relu() # d = 8.0
d.backward()

print(f"Custom Engine -> a.grad: {a.grad}, b.grad: {b.grad}, c.grad: {c.grad}")

# 2. PyTorch
ta = torch.tensor(2.0, requires_grad=True)
tb = torch.tensor(-3.0, requires_grad=True)
tc = torch.tensor(10.0, requires_grad=True)
td = ((ta * tb + tc) * 2.0).relu()
td.backward()

print(f"PyTorch       -> a.grad: {ta.grad.item()}, b.grad: {tb.grad.item()}, c.grad: {tc.grad.item()}")

# 3. Assert difference is < 1e-6
diff_a = abs(a.grad - ta.grad.item())
diff_b = abs(b.grad - tb.grad.item())
diff_c = abs(c.grad - tc.grad.item())
print(f"Max Absolute Difference: {max(diff_a, diff_b, diff_c)}")