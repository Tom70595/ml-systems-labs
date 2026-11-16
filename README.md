# ML Systems Labs

Hands-on labs from my Machine Learning Systems course at Vistula University, Warsaw. They focus on how ML models behave as systems: correctness, cost, reproducibility and deployment limits. All code is in Python.

## Labs

| Lab | Topic | What it covers |
| --- | --- | --- |
| 01 | Environment and first measurements | Pinned environment; logistic regression vs random forest on Breast Cancer Wisconsin; training time, inference latency, model size and peak memory; fit for cloud, edge, mobile and TinyML budgets |
| 02 | Neural network from scratch | Two-hidden-layer network on the digits dataset; forward pass, cross-entropy loss, hand-derived backpropagation; numerical gradient checking; learning-rate and width experiments; forward vs backward timing |
| 03 | Architectures and their costs | MLP, CNN and attention models in PyTorch on Fashion-MNIST; parameter counts; hand-calculated vs profiled FLOPs; latency at batch 1 and 128 |
| 04 | Data pipeline and quality | Reproducible, leakage-free pipeline on UCI Adult; schema and range validation; train-only imputation; duplicate and label-conflict audits; data hash and data card |

## Setup

Python 3.11.8 is used.

```
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
lab01/ ... lab04/
  report.md    goal, method, results, conclusions
  src/         code
  results/     plots, tables, saved outputs
```

Each lab's `report.md` explains the method and results. Datasets download automatically when a script runs.

## Reproducibility

- Fixed random seeds (42)
- Pinned dependency versions
- Timings reported as the median of repeated runs after a warm-up
