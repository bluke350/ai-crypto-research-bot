from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import accuracy_score  # type: ignore
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional dep
    RandomForestClassifier = None
    train_test_split = None
    accuracy_score = None
    joblib = None


def _collect_summary_files(artifacts_root: str) -> List[str]:
    out = []
    if not os.path.exists(artifacts_root):
        return out
    for d in os.listdir(artifacts_root):
        rd = os.path.join(artifacts_root, d)
        if not os.path.isdir(rd):
            continue
        sfn = os.path.join(rd, "summary.json")
        if os.path.exists(sfn):
            out.append(sfn)
    return out


def _load_features(summaries: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
    X = []
    y = []
    keys = set()
    rows = []
    for s in summaries:
        try:
            d = json.load(open(s, "r", encoding="utf-8"))
        except Exception:
            continue
        rows.append(d)
        keys.update(k for k, v in d.items() if isinstance(v, (int, float)))

    keys = sorted(keys)
    for d in rows:
        vec = [float(d.get(k) or 0.0) for k in keys]
        X.append(vec)
        y.append(1 if d.get("passed") else 0)
    return np.asarray(X, dtype=float), keys, y


def train_policy(artifacts_root: str = "experiments/artifacts", out_path: str = "models/policy.pkl") -> None:
    """Train a classifier from historical run `summary.json` files.

    Saves a joblib artifact containing {'model': model, 'features': keys, 'feature_importances': importances}
    Uses RandomForestClassifier for better non-linear separation.
    """
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn and joblib required to train policy. Install scikit-learn and joblib.")

    files = _collect_summary_files(artifacts_root)
    if len(files) < 5:
        raise RuntimeError("not enough historical runs to train policy (need >=5)")

    X, keys, y = _load_features(files)
    if X.shape[0] < 5:
        raise RuntimeError("insufficient numeric feature data to train")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds) if accuracy_score is not None else None
    importances = model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({"model": model, "features": keys, "feature_importances": importances}, out_path)
    print("trained policy saved to:", out_path, "val_acc:", acc)
    if importances is not None:
        print("feature importances:")
        for k, v in sorted(zip(keys, importances), key=lambda x: -x[1]):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-root", default="experiments/artifacts")
    p.add_argument("--out", default="models/policy.pkl")
    args = p.parse_args()
    train_policy(args.artifacts_root, args.out)
