## Lab 01 — Environment and First System Measurements

## 1. Goal

Set up a reproducible Python 3.11.8 environment with pinned dependencies,
train two baseline models (Logistic Regression and Random Forest) on the
Breast Cancer Wisconsin dataset, measure training time, single-sample
inference latency, model size, and peak memory, and decide where each
model could realistically be deployed (Cloud / Edge / Mobile / TinyML).

## 2. Method

- **Environment:** Python 3.11.8, isolated venv, dependencies pinned in `requirements.txt`.
  Measurements were collected inside a GitHub Codespaces Linux VM
  (see `lab01/results/versions.txt` for exact Python and library versions).
- **Dataset:** Breast Cancer Wisconsin, `load_breast_cancer(return_X_y=True)`;
  70/30 stratified split, `random_state=42`.
- **Models:**
  - `LogisticRegression(max_iter=1000, random_state=42)`
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Seeds:** `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`,
  `torch.cuda.manual_seed_all(42)`.
- **Training time:** 1 warm-up run, then 5 measured runs; median reported.
- **Single-sample inference:** 100 runs; median reported in milliseconds.
- **Model size:** serialized with `joblib.dump`; bytes and KB recorded.
- **Peak memory:** `psutil.Process().memory_info().rss` delta during
  training and during single-sample inference.

## 3. Results

### 3.1 Baseline accuracy

| Model | Test accuracy |
|---|---|
| Logistic Regression | 0.9474 |
| Random Forest | 0.9357 |

### 3.2 System measurements

| Model | Train median (s) | Infer median (ms) | Size (bytes) | Size (KB) | Peak train mem (MB) | Peak infer mem (MB) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.1506 | 0.1055 | 1103 | 1.08 | 0.02 | 0.00 |
| Random Forest | 0.1435 | 2.7666 | 290905 | 284.09 | 0.34 | 0.00 |

### 3.3 Deployment fit

Budgets (from the lab manual):

| Tier | Memory | Latency | Model size |
|---|---|---|---|
| Cloud | ≥ 1 GB | ≤ 100 ms | ≤ 500 MB |
| Edge | 256–1024 MB | ≤ 50 ms | ≤ 50 MB |
| Mobile | 64–256 MB | ≤ 20 ms | ≤ 10 MB |
| TinyML | ≤ 256 KB | ≤ 10 ms | ≤ 100 KB |

Fit per model:

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | ✅ PASS | ✅ PASS | ✅ PASS | ⚠️ Partial (model size and latency pass; runtime memory exceeds 256 KB because Python cannot run in ≤ 256 KB) |
| Random Forest | ✅ PASS | ✅ PASS | ✅ PASS | ❌ FAIL (model size 284 KB > 100 KB budget) |

**Logistic Regression.** The model itself is 1.08 KB and single-sample
inference latency is ~0.11 ms — both well inside every tier including TinyML.
The only blocker for TinyML is the runtime: executing a scikit-learn model
requires a Python interpreter whose baseline RSS is far above 256 KB.
A TinyML deployment would require exporting the model to a compact format
(e.g., ONNX → C) and executing it on a microcontroller runtime.

**Random Forest.** Same story as Logistic Regression for Cloud / Edge / Mobile:
size 284 KB and latency 2.77 ms are both comfortably inside those budgets.
For TinyML the model size alone (284 KB) is roughly 3× the 100 KB budget,
so it fails regardless of runtime. Random Forest is therefore best suited
to Cloud, Edge, or Mobile.

### 3.4 Notes on measurement honesty

The measured "peak memory" is the RSS delta across the `.fit()` or
`.predict()` call, not the entire Python process. The Python interpreter
baseline (around tens of MB) is not included in that delta, which is why
the values look very small. For deployment decisions, the interpreter
overhead must be added on top.

## 4. Three conclusions

1. **Model size and latency do not always favor the same model.** Logistic
   Regression is ~263× smaller (1.08 KB vs 284.09 KB) and ~26× faster at
   single-sample inference (0.11 ms vs 2.77 ms) than Random Forest, yet both
   are trained in essentially the same time (~0.15 s). For latency- or
   memory-constrained deployment, Logistic Regression is clearly preferable.

2. **Accuracy alone is a poor deployment criterion.** Logistic Regression
   actually scored slightly higher on the test set (0.9474 vs 0.9357) while
   being dramatically cheaper on every other axis. This dataset is nearly
   linearly separable, so a complex tree ensemble brings no accuracy benefit
   — a reminder that model selection must weigh accuracy *and* system cost.

3. **TinyML is a runtime problem, not just a model-size problem.** Even a
   1.08 KB Logistic Regression model cannot run on a 256 KB microcontroller
   as-is, because scikit-learn requires a Python interpreter. Meeting a
   TinyML budget requires exporting the model to a compact, dependency-free
   format (ONNX Runtime Mini, TFLite Micro, or hand-written C) — the model
   numbers alone are misleading.