# Lab 05: Frameworks and Automatic Differentiation

## Task 1, 2 & 3: Custom Autograd Engine
I implemented a scalar automatic-differentiation engine (`Value` class) from scratch. 
- **Task 1:** The `Value` node stores `data`, `grad`, `_prev` (parent nodes), and `_backward` (local derivative rule).
- **Task 2:** Implemented `__add__`, `__mul__`, `__pow__`, and `relu()`. These operations build a computation graph during the forward pass.
- **Task 3:** The `backward()` method performs a topological sort of the graph, sets the output gradient to `1.0`, and visits nodes in reverse order. It accumulates gradients using `+=` to correctly handle nodes with multiple children.

## Task 4: Gradient Verification
I verified the custom engine against PyTorch autograd on the expression `((a*b + c)*2).relu()`.
- **Custom Engine:** `a.grad = -6.0`, `b.grad = 4.0`, `c.grad = 2.0`
- **PyTorch:** `a.grad = -6.0`, `b.grad = 4.0`, `c.grad = 2.0`
- **Max Absolute Difference:** `0.0` (Passes the `< 1e-6` requirement).

## Task 5: Training a Tiny Model on make_moons
Using the custom engine, I trained a `2 -> 8 -> 1` MLP on the `make_moons` dataset for 50 epochs using SGD.
- **Training Time:** 2.8007 seconds.
- Because the engine is implemented in pure Python loops rather than C++ tensors, it is significantly slower than PyTorch, highlighting the overhead of Python in deep learning frameworks.

## Task 6: Eager vs. Compiled Execution in PyTorch
I compared Eager execution against Traced execution (`torch.jit.trace`) using a simple `Linear(100, 100)` model.
- **Eager Execution Time:** 0.9184 seconds
- **Traced Execution Time:** 0.0320 seconds
- **Speedup:** 28.71x faster.

**When compilation helps:**
Compilation (or tracing) helps by fusing operations (like combining matrix multiplication and addition) and removing Python overhead. It is most effective for large models with highly repetitive graph structures. However, it requires a warm-up time to build the graph and can struggle with dynamic control flow (like `if` statements that depend on data values).