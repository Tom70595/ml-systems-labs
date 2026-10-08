import os
import csv
import time
import random
import statistics
import joblib
import numpy as np
import psutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
except Exception:
    pass

RESULTS = "lab01/results"
os.makedirs(RESULTS, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

def make_linear():
    return LogisticRegression(max_iter=1000, random_state=42)

def make_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def time_training(factory, X, y, warmup=1, runs=5):
    for _ in range(warmup):
        factory().fit(X, y)
    times = []
    for _ in range(runs):
        m = factory()
        t0 = time.perf_counter()
        m.fit(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)

def time_single_inference(model, x_one, runs=100):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(x_one.reshape(1, -1))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times)

def peak_rss_mb(fn):
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss
    fn()
    after = proc.memory_info().rss
    return max(after - before, 0) / (1024 * 1024)

lin_train_med = time_training(make_linear, X_train, y_train)
linear = make_linear().fit(X_train, y_train)
lin_acc = accuracy_score(y_test, linear.predict(X_test))
x_one = X_test[0]
lin_inf_med = time_single_inference(linear, x_one)
lin_mem_train = peak_rss_mb(lambda: make_linear().fit(X_train, y_train))
lin_mem_inf = peak_rss_mb(lambda: linear.predict(x_one.reshape(1, -1)))
joblib.dump(linear, f"{RESULTS}/linear_model.joblib")
lin_bytes = os.path.getsize(f"{RESULTS}/linear_model.joblib")

rf_train_med = time_training(make_forest, X_train, y_train)
forest = make_forest().fit(X_train, y_train)
rf_acc = accuracy_score(y_test, forest.predict(X_test))
rf_inf_med = time_single_inference(forest, x_one)
rf_mem_train = peak_rss_mb(lambda: make_forest().fit(X_train, y_train))
rf_mem_inf = peak_rss_mb(lambda: forest.predict(x_one.reshape(1, -1)))
joblib.dump(forest, f"{RESULTS}/forest_model.joblib")
rf_bytes = os.path.getsize(f"{RESULTS}/forest_model.joblib")

with open(f"{RESULTS}/baseline_accuracy.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "test_accuracy"])
    w.writerow(["logistic_regression", f"{lin_acc:.4f}"])
    w.writerow(["random_forest", f"{rf_acc:.4f}"])

with open(f"{RESULTS}/measurements.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "model", "train_median_s", "infer_median_ms",
        "size_bytes", "size_kb", "peak_train_mb", "peak_infer_mb",
        "test_accuracy",
    ])
    w.writerow([
        "logistic_regression", f"{lin_train_med:.6f}", f"{lin_inf_med:.6f}",
        lin_bytes, f"{lin_bytes/1024:.2f}", f"{lin_mem_train:.2f}",
        f"{lin_mem_inf:.2f}", f"{lin_acc:.4f}",
    ])
    w.writerow([
        "random_forest", f"{rf_train_med:.6f}", f"{rf_inf_med:.6f}",
        rf_bytes, f"{rf_bytes/1024:.2f}", f"{rf_mem_train:.2f}",
        f"{rf_mem_inf:.2f}", f"{rf_acc:.4f}",
    ])

print("Done. See lab01/results/")