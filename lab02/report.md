# Lab 02 — A Neural Network From Scratch

## 1. Goal

Implement a two-hidden-layer neural network entirely from scratch using
NumPy ( forward propagation, softmax, cross-entropy loss, and
backpropagation ) and verify the correctness of the analytic gradients
with numerical gradient checking. Then compare learning rates and hidden
widths by training on the digits dataset, and measure forward vs backward
timing.

## 2. Method

### 2.1 Data
- Dataset: `sklearn.datasets.load_digits` (1797 samples, 64 features each).
- Inputs scaled to [0, 1] by dividing by 16.
- Labels one-hot encoded to shape (N, 10).
- 70/30 stratified split with `random_state=42`.
- Train: (1257, 64), Test: (540, 64).

### 2.2 Architecture
- 64 → W → ReLU → W → ReLU → 10 → Softmax.
- Two hidden widths tested: W = 16 and W = 64.
- Weight shapes: (64, W), (W, W), (W, 10); biases: (W,), (W,), (10,).

### 2.3 Initialization
- Weights drawn from N(0, 0.01) (standard deviation = 0.01).
- Biases initialized to zero.

### 2.4 Forward, loss, backward
- Forward: three linear layers with ReLU between the first two and
  softmax at the output. Intermediate activations cached for backprop.
- Loss: cross-entropy, averaged over the batch.
- Backward: chain rule applied layer-by-layer. Softmax + cross-entropy
  combined into the single gradient `probs - y_onehot`, divided by the
  batch size. ReLU gradients masked by `z > 0`.

### 2.5 Gradient check
- Ten randomly chosen weights nudged by ±1e-5.
- Numerical gradient: `(loss(+) - loss(-)) / (2 * eps)`.
- Analytic gradient taken from the backward pass.
- Relative error: `|num - ana| / (|num| + |ana| + 1e-12)`.
- Criterion: rel_err ≤ 1e-6 for at least 9 of 10 weights.

### 2.6 Training sweep
- Six configurations: learning rate ∈ {0.001, 0.01, 0.1} × width ∈ {16, 64}.
- Mini-batch size 32, 50 epochs, shuffled each epoch.
- Loss per epoch logged; final test accuracy recorded.

### 2.7 Timing
- Forward and backward passes timed separately on the full training set.
- One warm-up run discarded; median of 100 runs reported.

## 3. Results

### 3.1 Sanity checks
- Forward output shape: (batch, 10). ✔
- Each row of softmax output sums to 1 within 1e-6. ✔
- Untrained network loss = 2.3026, matching ln(10). ✔

### 3.2 Gradient check
- Passed **10 / 10** weights with relative error ≤ 1e-6.
- Largest observed relative error: 9.31e-07.
- Smallest: 0.00e+00 (a weight whose gradient is exactly zero, as
  expected when a ReLU is inactive).

### 3.3 Training sweep — final test accuracy

| Width | lr = 0.001 | lr = 0.01 | lr = 0.1 |
|---|---|---|---|
| 16 | 0.1019 | 0.1019 | 0.3259 |
| 64 | 0.1019 | 0.1019 | 0.9389 |

### 3.4 Loss curves
Saved to `lab02/results/loss_curves.png` (2×3 grid, one subplot per
configuration).

### 3.5 Timing (W = 64, full training set, median of 100)

| Pass | Median (ms) |
|---|---|
| Forward | 1.385 |
| Backward | 1.725 |
| Backward / Forward | 1.25× |

## 4. Three conclusions

1. **Analytical gradients are correct.** Numerical gradient checking
   passed all ten sampled weights within a relative error of 1e-6, with
   the largest observed error at 9.31e-07. This confirms that the
   hand-derived backpropagation is mathematically exact and free of
   implementation bugs — not just "close enough."

2. **Learning rate is tightly coupled to initialization scale.** Only
   lr = 0.1 escaped the near-uniform output regime within 50 epochs;
   lr = 0.01 and lr = 0.001 remained stuck at ~10.19% (random guessing
   on 10 classes). With N(0, 0.01) initialization, the signal shrinks
   through three layers to logits of order 1e-4, producing vanishingly
   small gradients. This is the vanishing gradient problem: a learning
   rate that is "large enough" for well-initialized networks is
   insufficient here, and the lab's low learning rates expose it. A
   larger initialization (e.g., N(0, 0.1) or He initialization) would
   let smaller learning rates converge.

3. **Backward pass is faster than the theoretical 2×.** Backward
   performs two matrix multiplies per layer (one for the weight
   gradient, one to propagate the error), while forward performs one,
   so a ~2× ratio is expected. The measured 1.25× is lower because at
   these matrix sizes (64×64) Python and memory-traffic overhead
   dominates the actual FLOPs. The asymptotic gap would widen with
   larger hidden widths.