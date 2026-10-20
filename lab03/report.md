# Lab 03 — Fashion-MNIST Architectures and Cost

## 1. Goal
The goal of this lab is to implement three neural network architectures of similar sizes (100k–150k parameters) — MLP, CNN, and Attention — and compare them not just on accuracy, but on system cost: FLOPs, parameter count, and inference latency. This highlights the core ML Systems trade-off: higher accuracy is meaningless if the model is too slow or too large to deploy.

## 2. Method

### 2.1 Data
- Dataset: Fashion-MNIST (`torchvision.datasets.FashionMNIST`).
- Normalization: `Normalize((0.2860,), (0.3530,))`.
- Split: 55,000 train, 5,000 validation, 10,000 official test. Batch size 128.
- Seed: 42.

### 2.2 Architectures (Adjusted for fair comparison)
All three models were adjusted to fit within the 100k–150k parameter range:
- **MLP**: Flatten → Linear(128) → ReLU → Linear(256) → ReLU → Linear(10). (136,074 params)
- **CNN**: Conv2d(1,16,3) → ReLU → Pool → Conv2d(16,32,3) → ReLU → Pool → Flatten → Linear(80) → ReLU → Linear(10). (131,130 params)
- **Attention**: 16 patches of 7×7 → Linear projection to 80 → 2 Transformer Encoder layers (d_model=80, 4 heads) → Mean pool → Linear(10). (108,970 params)

### 2.3 Training
- Optimizer: Adam, learning rate `1e-3`.
- Epochs: 5.
- Hardware: CPU (GitHub Codespace).

## 3. Results

### 3.1 Cost and Performance Table

| Model | Params | FLOPs (M) | Latency B=1 (ms) | Latency B=128 (ms) | Test Acc | Train Time (s) |
|---|---|---|---|---|---|---|
| MLP | 136,074 | 0.14 | 0.058 | 0.442 | 87.09% | 44.8 |
| CNN | 131,130 | 1.14 | 0.383 | 27.485 | **89.47%** | 132.2 |
| Attention | 108,970 | 0.06* | 1.086 | 13.821 | 84.38% | 115.9 |

*( *Note: `thop.profile` significantly undercounts FLOPs for PyTorch Transformers, as it does not trace the internal multi-head attention math. The true FLOP count is much higher. This is a known gap.)*

### 3.2 Hand-Calculated FLOPs vs. `thop`
The manual asks to hand-calculate one layer of each type.
- **Linear (MLP layer 1)**: `2 × 784 × 128 = 200,704 FLOPs`
- **Conv2d (CNN layer 1)**: `2 × 3 × 3 × 1 × 16 × 28 × 28 = 225,792 FLOPs`
- **Attention (QKV projection)**: `2 × 16 × 80 × 3 × 80 = 614,400 FLOPs`
- **Attention (scores)**: `2 × 16² × 80 = 40,960 FLOPs`

When comparing these hand calculations to `thop`, we see a significant gap. `thop` uses hooks and tracing to estimate FLOPs. It struggles with dynamic operations (like Transformers) and sometimes double-counts or misses connections. For the MLP and CNN, `thop`'s estimates (0.14M and 1.14M) align reasonably well with manual sums. For Attention, `thop` misses the bulk of the math entirely (reporting 0.06M), while the QKV projection alone is 0.6M FLOPs.

### 3.3 Plots
*(See `lab03/results/plot_accuracy_vs_params.png` and `plot_accuracy_vs_latency.png`)*

## 4. Three Conclusions

1. **The CNN is the most cost-effective architecture for this task.** It achieved the highest accuracy (89.47%) while maintaining a modest parameter count (131k). Although it has the highest FLOP count (1.14M) due to the convolution operations, its inference latency for a single image (0.38 ms) is highly acceptable for real-time systems. For image data, convolutional inductive biases (locality and translation invariance) are extremely powerful, allowing it to beat the other architectures with only 5 epochs of training.

2. **The Attention model is the least cost-effective here.** It was the slowest for batch size 1 (1.086 ms) and achieved the lowest accuracy (84.38%). Vision Transformers (ViTs) lack the built-in spatial biases of CNNs and typically require massive datasets (like ImageNet) and hundreds of epochs to match CNN performance. At this small scale (Fashion-MNIST, 55k images, 5 epochs), the Attention mechanism is computationally inefficient and underperforms. 

3. **The MLP is the fastest but fails on the accuracy-latency frontier.** The MLP is remarkably fast (0.058 ms latency) and trains very quickly (44.8s). However, it flattens the image into a 1D array, destroying all spatial structure. As a result, it stalls at 87.09% accuracy. While it is excellent for extremely latency-constrained environments (like TinyML), the CNN offers a much better trade-off for image classification, gaining +2.4% accuracy for only 0.3 ms of extra latency.