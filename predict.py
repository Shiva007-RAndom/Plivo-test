"""Final EOT predictor: loads a SAVED model (see fit_final_models.py) and
scores pauses in a data_dir it has never trained on. Does not fit/retrain
here -- that's the whole point of shipping a saved model instead of the
train.py dev-loop script.

Matches baseline.py's interface exactly:
    python predict.py --data_dir eot_data/english --out predictions.csv

Only reads turn_id, audio_file, pause_index, pause_start from labels.csv --
label/pause_end (if present) are ignored, since a live pause only has causal
information available (see features.py's causality note).
"""
import argparse
import csv
import os

import joblib
import numpy as np

from features import load_wav
from train import extract_features

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")


def load_model_for(data_dir):
    """Pick the model matching this data_dir's language by folder name;
    fall back to the pooled model for an unrecognized folder so this never
    hard-fails on a genuinely new/unseen directory name."""
    name = os.path.basename(os.path.normpath(data_dir)).lower()
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        path = os.path.join(MODEL_DIR, "combined.joblib")
    return joblib.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    clf = load_model_for(args.data_dir)

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    keys, X = [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        keys.append((r["turn_id"], r["pause_index"]))
    X = np.array(X)

    p = clf.predict_proba(X)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
