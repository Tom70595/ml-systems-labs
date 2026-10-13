import os
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============ Task 1: Data ============
def load_data():
    X, y = load_digits(return_X_y=True)          # (1797, 64), (1797,)
    X = X / 16.0                                  # scale to [0, 1]
    y_onehot = np.eye(10)[y]                      # one-hot
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.30, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ============ Task 2: Network init ============
def init_network(W, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.01, (64, W)),
        "b1": np.zeros(W),
        "W2": rng.normal(0, 0.01, (W, W)),
        "b2": np.zeros(W),
        "W3": rng.normal(0, 0.01, (W, 10)),
        "b3": np.zeros(10),
    }


# ============ Task 3: Forward pass ============
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)   # numerical stability
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    probs = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2,
             "z3": z3, "probs": probs}
    return probs, cache


# ============ Task 4: Cross-entropy loss ============
def cross_entropy(probs, y_onehot):
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


# ============ Task 5: Backward pass ============
def backward(params, cache, y_onehot):
    X = cache["X"]
    z1, a1 = cache["z1"], cache["a1"]
    z2, a2 = cache["z2"], cache["a2"]
    probs = cache["probs"]
    m = X.shape[0]

    dz3 = (probs - y_onehot) / m                # softmax + CE shortcut
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (z2 > 0)                         # ReLU derivative
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {"W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3}


# ============ Task 5: Gradient check ============
def gradient_check(params, X, y_onehot, num_checks=10, eps=1e-5, seed=0):
    probs, cache = forward(params, X)
    grads = backward(params, cache, y_onehot)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(num_checks):
        key = rng.choice(["W1", "W2", "W3"])
        idx = tuple(int(rng.integers(0, s)) for s in params[key].shape)
        orig = params[key][idx]

        params[key][idx] = orig + eps
        loss_p = cross_entropy(forward(params, X)[0], y_onehot)

        params[key][idx] = orig - eps
        loss_m = cross_entropy(forward(params, X)[0], y_onehot)

        params[key][idx] = orig
        num_grad = (loss_p - loss_m) / (2 * eps)
        ana_grad = grads[key][idx]
        denom = abs(num_grad) + abs(ana_grad) + 1e-12
        rel_err = abs(num_grad - ana_grad) / denom
        results.append((key, idx, num_grad, ana_grad, rel_err))
    return results


# ============ Task 6: Training ============
def train(X_train, y_train, X_test, y_test, W, lr,
          epochs=50, batch_size=32, seed=42):
    params = init_network(W, seed=seed)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    losses = []

    for epoch in range(epochs):
        perm = rng.permutation(n)
        Xs, ys = X_train[perm], y_train[perm]
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            Xb, yb = Xs[i:i+batch_size], ys[i:i+batch_size]
            probs, cache = forward(params, Xb)
            loss = cross_entropy(probs, yb)
            grads = backward(params, cache, yb)
            for k in params:
                params[k] -= lr * grads[k]
            epoch_loss += loss
            n_batches += 1
        losses.append(epoch_loss / n_batches)

    probs_test, _ = forward(params, X_test)
    preds = np.argmax(probs_test, axis=1)
    true = np.argmax(y_test, axis=1)
    acc = float(np.mean(preds == true))
    return losses, acc, params


# ============ Task 7: Timing ============
def time_forward_backward(params, X, y_onehot, n_runs=100):
    # warm-up
    probs, cache = forward(params, X)
    backward(params, cache, y_onehot)

    fwd, bwd = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        probs, cache = forward(params, X)
        t1 = time.perf_counter()
        backward(params, cache, y_onehot)
        t2 = time.perf_counter()
        fwd.append((t1 - t0) * 1000.0)
        bwd.append((t2 - t1) * 1000.0)
    return float(np.median(fwd)), float(np.median(bwd))


# ============ Main ============
if __name__ == "__main__":
    os.makedirs("lab02/results", exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- Task 3 sanity ---
    params = init_network(W=16)
    probs, _ = forward(params, X_train[:5])
    print(f"Forward output shape: {probs.shape}")
    print(f"Row sums: {probs.sum(axis=1)}")

    # --- Task 4 sanity ---
    probs_full, _ = forward(params, X_train)
    loss0 = cross_entropy(probs_full, y_train)
    print(f"Untrained loss: {loss0:.4f}  (expected ~{np.log(10):.4f})")

    # --- Task 5 gradient check ---
    Xs, ys = X_train[:32], y_train[:32]
    results = gradient_check(params, Xs, ys, num_checks=10)
    print("\nGradient check:")
    n_pass = 0
    for key, idx, ng, ag, err in results:
        ok = err <= 1e-6
        n_pass += int(ok)
        print(f"  {key}{idx}: num={ng: .6e}  ana={ag: .6e}  "
              f"rel_err={err:.2e}  {'PASS' if ok else 'FAIL'}")
    print(f"Passed {n_pass}/10  (need >= 9)")

    # --- Task 6 training sweep ---
    print("\nTraining sweep:")
    configs = [(16, 0.001), (16, 0.01), (16, 0.1),
               (64, 0.001), (64, 0.01), (64, 0.1)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (W, lr) in zip(axes.flatten(), configs):
        losses, acc, _ = train(X_train, y_train, X_test, y_test, W, lr)
        ax.plot(losses)
        ax.set_title(f"W={W}, lr={lr}\nfinal acc={acc:.4f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        print(f"  W={W:2d}, lr={lr}: final acc = {acc:.4f}")
    plt.tight_layout()
    plt.savefig("lab02/results/loss_curves.png", dpi=120)
    print("Saved lab02/results/loss_curves.png")

    # --- Task 7 timing ---
    params = init_network(W=64)
    fwd_ms, bwd_ms = time_forward_backward(params, X_train, y_train, n_runs=100)
    print(f"\nTiming (W=64, full train set):")
    print(f"  Forward median:  {fwd_ms:.3f} ms")
    print(f"  Backward median: {bwd_ms:.3f} ms")
    print(f"  Backward / forward ratio: {bwd_ms/fwd_ms:.2f}x")