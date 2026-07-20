"""Skeleton: prosodic features + classifier. Runs as-is, scores poorly ON
PURPOSE. Your hour goes into extract_features() and what you learn from
your errors.

    python train.py --data_dir eot_data/english --out predictions.csv

Ideas worth testing (this is the assignment, not a checklist):
  - F0 slope over the last voiced region (statements fall, continuations
    often stay level or rise)
  - final-syllable lengthening: last voiced stretch duration vs the
    speaker's average
  - energy decay rate into the pause
  - speaking-rate context, position of the pause within the turn so far
  - anything you discover by LISTENING to your misclassified pauses
"""
import argparse
import csv
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from features import (
    load_wav, speech_before, frame_energy_db, f0_contour, HOP_MS,
    f0_slope_last_voiced, final_lengthening_ratio, energy_decay_rate,
    zscore, energy_baseline, pitch_baseline, smooth_voicing_gaps,
)


def extract_features(x, sr, pause_start):
    """Features from audio STRICTLY BEFORE pause_start."""
    seg = speech_before(x, sr, pause_start, window_s=2.5)
    if len(seg) < sr // 10:
        return np.zeros(8, dtype=np.float32)
    e = frame_energy_db(seg, sr)
    f0 = smooth_voicing_gaps(f0_contour(seg, sr))
    voiced = f0[f0 > 0]
    hop_s = HOP_MS / 1000.0
    energy_now = e[-5:].mean()
    pitch_now = voiced[-3:].mean() if len(voiced) >= 3 else 0.0

    # normalize against this turn's own causal history (no speaker id is
    # available -- audio_file/turn_id is the closest proxy we have)
    hist = x[: int(pause_start * sr)]
    e_mean, e_std = energy_baseline(hist, sr)
    p_mean, p_std = pitch_baseline(hist, sr)

    return np.array([
        energy_now,                                       # energy right before the pause
        pitch_now,                                         # final pitch
        len(seg) / sr,                                    # how much speech context we had
        f0_slope_last_voiced(f0, hop_s),                  # F0 trend into the pause
        final_lengthening_ratio(f0),                      # final voiced stretch vs typical
        energy_decay_rate(e, hop_s),                      # energy decay rate into the pause
        zscore(energy_now, e_mean, e_std),                # energy vs this turn's own baseline
        zscore(pitch_now, p_mean, p_std) if pitch_now > 0 else 0.0,  # pitch vs baseline
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
    X, y, groups = np.array(X), np.array(y), np.array(groups)

    # out-of-fold predictions: every turn is scored by a model that never
    # trained on it, so what we write out (and what score.py grades) reflects
    # held-out performance instead of the model grading its own training data
    n_splits = max(2, min(args.folds, len(set(groups))))
    oof = np.zeros(len(y), dtype=np.float64)
    fold_acc = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=5,
            class_weight="balanced", random_state=0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        fold_acc.append(clf.score(X[te], y[te]))
    print(f"held-out turn accuracy ({n_splits}-fold mean): {np.mean(fold_acc):.3f} "
          f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, oof):
            w.writerow([tid, pi, f"{p:.4f}"])
    print(f"wrote {len(keys)} out-of-fold predictions -> {args.out}")
    print("NOTE for your final predict.py: it must load a SAVED model (fit on "
          "all labeled data) and predict on genuinely unseen data -- this "
          "script's per-fold models are for honest evaluation only.")


if __name__ == "__main__":
    main()
