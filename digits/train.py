"""Handwritten-digit classification on scikit-learn's bundled 8x8 digits dataset (1797 images)."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_split(seed: int = 0):
    d = load_digits()
    return train_test_split(d.data / 16.0, d.target, test_size=0.25, stratify=d.target, random_state=seed)


MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000),
    "svm_rbf": lambda: SVC(gamma=0.02, C=10),
    "mlp": lambda: MLPClassifier((64,), max_iter=500, random_state=0),
}


def evaluate_all(seed: int = 0) -> dict:
    Xtr, Xte, ytr, yte = get_split(seed)
    out = {}
    for name, make in MODELS.items():
        m = make().fit(Xtr, ytr)
        out[name] = round(accuracy_score(yte, m.predict(Xte)), 4)
    return out


def worst_confusions(seed: int = 0, top: int = 3):
    Xtr, Xte, ytr, yte = get_split(seed)
    pred = MODELS["svm_rbf"]().fit(Xtr, ytr).predict(Xte)
    cm = confusion_matrix(yte, pred)
    np.fill_diagonal(cm, 0)
    flat = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if cm[i, j]]
    return sorted(flat, reverse=True)[:top]  # (count, true, predicted)
