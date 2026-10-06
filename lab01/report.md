# Lab 01 — Environment and First System Measurements

## 1. Goal

Set up a reproducible Python 3.11.8 environment with pinned dependencies,
train two baseline models (Logistic Regression and Random Forest) on the
Breast Cancer Wisconsin dataset, measure training time, single-sample
inference latency, model size, and peak memory, and decide where each
model could realistically be deployed (Cloud / Edge / Mobile / TinyML).

## 2. Method

- **Environment:** Python 3.11.8, isolated venv, dependencies pinned in `requirements.txt`.
- **Dataset:** `load_breast_cancer(return_X_y=True)`; 70/30 stratified split, `random_state=42`.
- **Models:**
  - `LogisticRegression(max_iter=1000, random_state=42)`
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Seeds:** `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
- **Training time:** 1 warm-up + 5 runs, median reported.
- **Single-sample inference:** 100 runs, median reported (ms).
- **Model size:** `joblib.dump`, bytes and KB recorded.
- **Peak memory:** `psutil.Process().memory_info().rss` delta during training and inference.

## 3. Results

### 3.1 Baseline accuracy

| Model | Test accuracy |
|---|---|
| Logistic Regression | (fill in) |
| Random Forest | (fill in) |

### 3.2 System measurements

| Model | Train median (s) | Infer median (ms) | Size (bytes) | Size (KB) | Peak train mem (MB) | Peak infer mem (MB) |
|---|---|---|---|---|---|---|
| Logistic Regression | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |
| Random Forest | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |

### 3.3 Deployment fit

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | (fill in) | (fill in) | (fill in) | (fill in) |
| Random Forest | (fill in) | (fill in) | (fill in) | (fill in) |

*(Justify each decision in prose, referring to the measured numbers.)*

## 4. Three conclusions

1. (Write your first conclusion in your own words.)
2. (Write your second conclusion in your own words.)
3. (Write your third conclusion in your own words.)
