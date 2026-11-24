# Thesis Project Proposal: AgriVision

**Student Name:** Tinomudaishe Tom
**Student ID:** 70595
**Degree:** Computer Engineering

## 1. Problem Statement
Farmers in remote or rural areas often lack reliable internet connectivity, making cloud-based AI solutions for crop disease diagnosis impractical. Furthermore, running heavy, state-of-the-art computer vision models locally on inexpensive edge devices (like budget smartphones or Raspberry Pis) is difficult due to strict limits on memory, processing power, and battery life. 

This project aims to bridge this gap by developing an offline, highly optimized image classification system that can run directly on resource-constrained hardware.

## 2. Dataset
**PlantVillage Dataset:** A widely used computer vision dataset containing over 50,000 images of healthy and diseased crop leaves across 38 different classes (species and disease combinations). 

## 3. Model Architectures
To demonstrate the trade-offs between accuracy and computational efficiency:
- **Teacher/Baseline Model:** MobileNetV2 (a standard, moderately sized CNN pre-trained on ImageNet).
- **Student Model:** A custom lightweight CNN with significantly fewer parameters.

## 4. Deployment Constraints
The final optimized model must meet the following edge-computing constraints:
- **Latency:** ≤ 50 ms per inference (on a CPU).
- **Size:** ≤ 5 MB on disk.

## 5. Optimization Techniques
To meet the deployment constraints while preserving accuracy, the following techniques will be applied and evaluated:
1. **Knowledge Distillation:** Transferring the "soft predictions" from the large Teacher model to the smaller Student model.
2. **Structured Pruning:** Removing entire convolutional filters based on L2 norm to reduce actual compute cost (not just sparsity).
3. **INT8 Post-Training Quantization:** Converting FP32 weights to INT8 to reduce the model size by approximately 4x and speed up CPU inference.

## 6. Evaluation Metrics
The project will compare the baseline, pruned, distilled, and quantized models on:
- Classification Accuracy (Target: within 2% of baseline)
- Inference Latency (ms)
- Model Size (MB)
- Peak Memory Consumption (RAM)
- Energy Efficiency (estimated via CPU utilization)

## 7. Technologies
- **Framework:** Python, PyTorch, Torchvision
- **MLOps:** MLflow for experiment tracking and reproducibility (from Lab 6)
- **Profiling:** `torch.profiler` and `psutil` for bottleneck analysis
- **Version Control:** Git/GitHub