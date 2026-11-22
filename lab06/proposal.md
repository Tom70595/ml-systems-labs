# Individual Project Proposal — Lab 6

**Student:**  <TINOMUDAISHE TOM>
**Student ID:** <70595>

## Problem
Deploy a lightweight image classifier for Fashion-MNIST (10-class clothing items) on a
resource-constrained edge device with strict latency and memory budgets.

## Dataset
Fashion-MNIST: 60,000 train + 10,000 test 28x28 grayscale images. Split: 55,000 train /
5,000 validation / 10,000 test (seed 42), normalization mean 0.2860, std 0.3530.

## Model
Small CNN (131k params, ~0.5 MB FP32) from Lab 3, which reached 89.47% test accuracy in
5 epochs — the best cost/accuracy trade-off among CNN, MLP, and Attention baselines.

## Success Metric
Validation accuracy ≥ 89.0% (within 0.5% of the FP32 baseline) after optimization.

## Deployment Constraints
- Inference latency ≤ 50 ms per image (batch 1, CPU)
- Model size ≤ 5 MB
- Current baseline: 0.38 ms latency, ~0.5 MB → ample headroom

## Planned Optimization
Post-training **INT8 dynamic quantization** (`torch.quantization.quantize_dynamic`).
Rationale: the CNN is dominated by Linear + Conv2d weights; quantization is the lowest-
effort method that preserves the existing training pipeline, should reduce model size by
~4x (0.5 MB → ~0.13 MB) and can cut CPU latency. I will compare FP32 vs INT8 on
accuracy, latency, and size and document the trade-off.

## Deliverables
- Optimized model artifact + evaluation script
- Accuracy/latency/size comparison table
- Reproducible config + checkpoint