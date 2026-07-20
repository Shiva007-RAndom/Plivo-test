"""Audio utilities for the EOT assignment.

These are UTILITIES, not features. Turning them into informative features
(slopes, ratios, statistics over time) is your job.

Causality reminder: for a pause at `pause_start`, you may only touch
audio[0 : pause_start]. Note that `pause_end` is FUTURE information for a
hold pause — using it (e.g., pause duration) in features is a violation.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=1.5):
    """The last `window_s` seconds of audio strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    """Short-time energy per frame, in dB."""
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Fundamental frequency of one frame via autocorrelation.

    Returns 0.0 for unvoiced/silent frames.
    """
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    """Per-frame F0 (Hz), 0.0 where unvoiced. Longer frames help pitch."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


def _linreg_slope(values, hop_s):
    """Least-squares slope of `values` over uniformly spaced steps of hop_s."""
    n = len(values)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64) * hop_s
    t_mean = t.mean()
    v = np.asarray(values, dtype=np.float64)
    denom = float(np.sum((t - t_mean) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum((t - t_mean) * (v - v.mean())) / denom)


def smooth_voicing_gaps(f0, max_gap=1):
    """Fill isolated unvoiced gaps of up to `max_gap` frames that are
    surrounded by voiced frames, by linear interpolation between the
    neighboring voiced values.

    A single noisy autocorrelation frame (voicing dips just under threshold)
    shouldn't fragment an otherwise continuous voiced region -- that
    fragmentation is what starves the run-based features (F0 slope, final
    lengthening) of signal right where they matter most: the trailing edge
    of speech into a pause.
    """
    f0 = np.array(f0, dtype=np.float32, copy=True)
    n = len(f0)
    i = 0
    while i < n:
        if f0[i] == 0:
            j = i
            while j < n and f0[j] == 0:
                j += 1
            gap_len = j - i
            if 0 < i and j < n and gap_len <= max_gap and f0[i - 1] > 0 and f0[j] > 0:
                for k in range(i, j):
                    t = (k - i + 1) / (gap_len + 1)
                    f0[k] = f0[i - 1] * (1 - t) + f0[j] * t
            i = j
        else:
            i += 1
    return f0


def voiced_runs(f0):
    """Contiguous (start, end) frame-index ranges where f0 > 0, end-exclusive."""
    runs = []
    start = None
    for i, val in enumerate(f0):
        if val > 0:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(f0)))
    return runs


def f0_slope_last_voiced(f0, hop_s, min_frames=3):
    """Hz/s slope of F0 over the last contiguous voiced run.

    Statements tend to fall into the pause; continuations often stay level
    or rise. 0.0 if there's no voiced run of at least `min_frames` frames.
    """
    runs = voiced_runs(f0)
    if not runs:
        return 0.0
    start, end = runs[-1]
    if end - start < min_frames:
        return 0.0
    return _linreg_slope(f0[start:end], hop_s)


def final_lengthening_ratio(f0):
    """Duration of the last voiced run vs the mean duration of earlier voiced
    runs in the same window -- a cue for final-syllable lengthening.

    Returns 1.0 (neutral) when there's fewer than two runs to compare.
    """
    runs = voiced_runs(f0)
    if len(runs) < 2:
        return 1.0
    durations = [end - start for start, end in runs]
    prior_mean = float(np.mean(durations[:-1]))
    if prior_mean <= 0:
        return 1.0
    return float(durations[-1] / prior_mean)


def energy_decay_rate(e, hop_s, tail_s=0.5):
    """dB/s slope of short-time energy over the last `tail_s` seconds.

    Negative values mean energy is falling off into the pause.
    """
    n_tail = max(2, int(round(tail_s / hop_s)))
    tail = e[-n_tail:]
    return _linreg_slope(tail, hop_s)


def zscore(value, mean, std, eps=1e-6):
    """Standard score of `value` against a (mean, std) baseline."""
    return float((value - mean) / (std + eps))


def energy_baseline(x, sr):
    """Mean/std short-time energy (dB) over the given audio."""
    e = frame_energy_db(x, sr)
    if len(e) == 0:
        return 0.0, 0.0
    return float(e.mean()), float(e.std())


def pitch_baseline(x, sr, frame_ms=40, hop_ms=HOP_MS):
    """Mean/std voiced F0 (Hz) over the given audio."""
    f0 = smooth_voicing_gaps(f0_contour(x, sr, frame_ms=frame_ms, hop_ms=hop_ms))
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return 0.0, 0.0
    return float(voiced.mean()), float(voiced.std())
