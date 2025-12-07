#!/usr/bin/env python3
"""Train a minimal scikit-learn logistic regression model and save to `models/opportunity.pkl`.

This is a convenience script to create a simple fallback model for testing the online update flow.
"""
from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    # synthetic dataset
    X = np.vstack([
        np.random.normal(loc=-1.0, scale=0.5, size=(200, 3)),
        np.random.normal(loc=1.0, scale=0.5, size=(200, 3)),
    ])
    y = np.hstack([np.zeros(200), np.ones(200)])

    # Use SGDClassifier with log loss to support partial_fit (online updates)
    clf = SGDClassifier(loss='log_loss', max_iter=1000)
    # partial_fit requires classes on first call
    classes = np.unique(y)
    clf.partial_fit(X[:10], y[:10], classes=classes)
    # do a few epochs of partial_fit to initialize
    for _ in range(5):
        clf.partial_fit(X, y)

    out = MODEL_DIR / "opportunity.pkl"
    with open(out, 'wb') as f:
        pickle.dump(clf, f)
    print('Saved fallback model to', out)


if __name__ == '__main__':
    main()
