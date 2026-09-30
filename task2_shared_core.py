from pathlib import Path

import numpy as np

DATA_DIR = Path(".")
N_CLASSES = 10

# Data Load
missing = [f"{s}_{p}.csv" for s in ("train", "test") for p in ("in", "out") if not (DATA_DIR / f"{s}_{p}.csv").exists()]
assert not missing, f"Missing {missing}: put the data files in the same folder as this notebook."


def load_split(split):
    X = np.loadtxt(DATA_DIR / f"{split}_in.csv", delimiter=",")
    y = np.loadtxt(DATA_DIR / f"{split}_out.csv", delimiter=",").astype(int)
    return X, y


X_train, y_train = load_split("train")
X_test, y_test = load_split("test")


# Perceptron Core
def add_bias(X):
    return np.hstack([X, np.ones((len(X), 1))])


def targets(y):
    T = -np.ones((len(y), N_CLASSES))
    T[np.arange(len(y)), y] = 1
    return T


def predict(W, X):
    return (add_bias(X) @ W).argmax(axis=1)


def accuracy(W, X, y):
    return (predict(W, X) == y).mean()


def perceptron_loss(W, X, y, rule="ovr"):
    A = add_bias(X) @ W
    if rule == "ovr":
        return np.maximum(0, -targets(y) * A).sum(axis=1).mean()
    return (A.max(axis=1) - A[np.arange(len(y)), y]).mean()


def update_direction(A, y, rule="ovr"):
    if rule == "ovr":
        T = targets(y)
        return T * (T * A <= 0)
    pred = A.argmax(axis=1)
    wrong = (pred != y).astype(float)
    M = np.zeros_like(A)
    M[np.arange(len(y)), y] += wrong
    M[np.arange(len(y)), pred] -= wrong
    return M


def train_perceptron(X, y, lr=1.0, init_std=0.0, rule="ovr", batch_size=32, max_epochs=200, seed=42,
                     X_eval=None, y_eval=None):
    rng = np.random.default_rng(seed)
    Xb = add_bias(X)
    W = init_std * rng.standard_normal((Xb.shape[1], N_CLASSES))
    hist = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}

    def record():
        hist["train_acc"].append(accuracy(W, X, y))
        hist["train_loss"].append(perceptron_loss(W, X, y, rule))
        if X_eval is not None:
            hist["test_acc"].append(accuracy(W, X_eval, y_eval))
            hist["test_loss"].append(perceptron_loss(W, X_eval, y_eval, rule))

    record()
    converged_at = None
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(X))
        n_mistakes = 0
        for start in range(0, len(X), batch_size):
            b = order[start:start + batch_size]
            M = update_direction(Xb[b] @ W, y[b], rule)
            W += lr * Xb[b].T @ M / len(b)
            n_mistakes += (M != 0).any(axis=1).sum()
        record()
        if n_mistakes == 0:
            converged_at = epoch - 1
            break
    for k, v in hist.items():
        hist[k] = np.array(v + v[-1:] * (max_epochs + 1 - len(v)))
    return W, hist, converged_at


DEFAULT = dict(lr=1.0, init_std=0.01, rule="ovr", batch_size=32, max_epochs=200)
