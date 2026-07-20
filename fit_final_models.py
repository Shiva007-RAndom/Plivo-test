"""One-time step: fit the final model(s) on ALL labeled data and save them
for predict.py to load. This is NOT the dev-loop -- that's train.py, which
uses GroupKFold cross-validation to give an honest, held-out performance
estimate (see RUNLOG.md). This script produces the model that actually ships:
once the recipe (features + hyperparameters) is validated via train.py, the
shipped model is refit on every labeled pause available, per the standard
"validate with CV, ship trained-on-everything" practice.

Saves:
    models/english.joblib   -- fit on all of eot_data/english
    models/hindi.joblib     -- fit on all of eot_data/hindi
    models/combined.joblib  -- fit on both pooled, fallback for an unrecognized
                                --data_dir name in predict.py

    python fit_final_models.py
"""
import csv
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from features import load_wav
from train import extract_features

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

# same recipe validated in RUNLOG.md RUN 7/8 via GroupKFold OOF evaluation
RF_PARAMS = dict(n_estimators=100, max_depth=3, min_samples_leaf=5,
                  class_weight="balanced", random_state=0)


def build_dataset(data_dir):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    cache = {}
    X, y = [], []
    for r in rows:
        path = os.path.join(data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
    return np.array(X), np.array(y)


def fit(X, y):
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X, y)
    return clf


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    datasets = {
        "english": build_dataset(os.path.join(ROOT, "eot_data/english")),
        "hindi": build_dataset(os.path.join(ROOT, "eot_data/hindi")),
    }
    for name, (X, y) in datasets.items():
        clf = fit(X, y)
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(clf, path)
        print(f"saved {path} ({len(y)} pauses, {y.sum()} eot / {len(y)-y.sum()} hold)")

    X_all = np.concatenate([X for X, y in datasets.values()])
    y_all = np.concatenate([y for X, y in datasets.values()])
    clf_combined = fit(X_all, y_all)
    path = os.path.join(MODEL_DIR, "combined.joblib")
    joblib.dump(clf_combined, path)
    print(f"saved {path} ({len(y_all)} pauses, pooled fallback)")


if __name__ == "__main__":
    main()
