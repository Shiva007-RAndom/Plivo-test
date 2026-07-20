## RUN 1

changes -
added _linreg_slope (helper), voiced_runs, and three new causal utilities: f0_slope_last_voiced, final_lengthening_ratio, energy_decay_rate. Each operates only on already-computed f0/e arrays derived from the causal speech_before window — no future audio is touched.

  mean response delay : 1220 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.45, delay=1100 ms

  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.45, delay=650 ms

  ## RUN 2

changes-
added zscore, energy_baseline, pitch_baseline (mean/std of energy and voiced pitch over any causal audio span). extract_features now also computes each pause's energy/pitch relative to a z-scored baseline built from x[:pause_start] (the same file's own history up to that point), giving 8 features total. Fallback zero-vector length updated to match.

  mean response delay : 1215 ms   <-- your score, lower is better
  interrupted turns   : 4.0%
  operating point     : threshold=0.45, delay=1100 ms

    mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms


  ## RUN 3

changes -
train.py previously reported held-out accuracy on a single 75/25 split but then refit the classifier on ALL data before writing predictions, so the scores being fed to score.py were in-sample (the model grading data it had memorized). Replaced this with grouped 5-fold cross-validation (GroupKFold, split by turn_id so a turn never straddles train/test): every row's p_eot in the output CSV now comes from a model that never trained on that turn (out-of-fold predictions). This makes the score.py numbers below the honest, held-out ones.

english:
  mean response delay : 1245 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.45, delay=1100 ms

hindi:
  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms

## RUN 4

changes -
error analysis on English's out-of-fold predictions showed false cutoffs were rare (7/148 holds) but 29/100 true EOTs scored below threshold and got forced to the 1.6s timeout -- the model was too conservative, not trigger-happy. Added smooth_voicing_gaps() to bridge isolated 1-frame voicing dropouts (features.py + train.py), on the hypothesis that a single noisy frame was fragmenting the trailing voiced run. Follow-up check showed that hypothesis was wrong for English: 0/63 low-scoring pauses had zero voicing or a too-short run -- plenty of signal was already present, the linear model just wasn't using it well. Side-by-side tested LogisticRegression (with/without StandardScaler), RandomForestClassifier, and GradientBoostingClassifier on identical features/GroupKFold splits: scaling and both nonlinear models beat the unscaled linear one on English, RandomForest was best there and didn't regress Hindi. Swapped the classifier in train.py from LogisticRegression to RandomForestClassifier (n_estimators=300, max_depth=4, min_samples_leaf=5, class_weight="balanced").

english:
  AUC                 : 0.598
  mean response delay : 1185 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.4, delay=1100 ms

hindi:
  AUC                 : 0.685
  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms

## RUN 5

changes -
grid-searched RandomForestClassifier hyperparameters (n_estimators x max_depth x min_samples_leaf) on English using the same GroupKFold OOF harness, optimizing for English's delay@5%-cutoff, then re-checked the winner on Hindi to confirm no regression before adopting it for real. Winner: n_estimators=100 (down from the guessed 300), max_depth=4, min_samples_leaf=5 -- same depth/leaf as before, just fewer trees. Updated train.py.

english:
  AUC                 : 0.603
  mean response delay : 1150 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.45, delay=1000 ms

hindi:
  AUC                 : 0.680
  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms

## RUN 6 (reverted)

changes -
tried adding context features: speaking_rate (voiced-run onsets/sec, a syllable-rate proxy), pause_index (ordinal position of this pause within the turn), and pause_start (elapsed turn time so far) -- all fully causal, no future leakage. Result on English: AUC roughly flat (0.603 -> 0.601) but delay got WORSE (1150ms -> 1264ms). Ablated each feature individually (and at max_depth=4 and 6): every single addition, alone or combined, made English's delay worse than the plain 8-feature baseline -- no combination beat 1150ms. Likely cause: only 100 turns / 248 pauses in English, so 3 extra dimensions added variance a tree model latches onto without enough data to generalize from. Hindi's AUC did improve with all 3 (0.680 -> 0.701) but its delay stayed pinned at 850ms (consistent with the suspected data floor from RUN 5's discussion), so no operational gain there either. Reverted extract_features() and removed the now-unused speaking_rate() from features.py -- train.py/features.py are back to the RUN 5 state. Keeping this logged so the idea isn't retried blind.

english (unchanged from RUN 5):
  AUC                 : 0.603
  mean response delay : 1150 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.45, delay=1000 ms

hindi (unchanged from RUN 5):
  AUC                 : 0.680
  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms

## RUN 7

changes -
after RUN 6 concluded prosody separation looked weak, tried a few not-yet-tested ideas rather than stop: (a) a trailing_silence feature (seconds since energy last exceeded window-max-20dB) -- made English worse, same overfitting pattern as RUN 6, dropped. (b) widened the look-back window in speech_before() from 1.5s to 2.5s, keeping the same 8 features -- this clearly helped (English AUC 0.603 -> 0.641) even before retuning. (c) re-ran the RandomForest hyperparameter grid at window=2.5s (still causal, just more history): best was n_estimators=100, max_depth=3, min_samples_leaf=5 (shallower than before -- more context per feature apparently needs less tree depth to exploit). Checked on Hindi before adopting: no regression, actually improved too. Updated train.py: window_s=1.5->2.5 in extract_features, max_depth=4->3 in the RandomForestClassifier.

english:
  AUC                 : 0.652
  mean response delay : 1120 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.45, delay=1000 ms

hindi:
  AUC                 : 0.691
  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms

## RUN 8 (no change -- confirmed current model is best)

changes -
before settling on RandomForest, compared it against AdaBoostClassifier, XGBoost, CatBoost, and GradientBoostingClassifier (27-combo hyperparameter search: n_estimators x max_depth x learning_rate), all on the same 8 features + window=2.5s + GroupKFold OOF harness. Tuned GradientBoosting edged out RandomForest on English delay by 8ms (1112ms vs 1120ms) but with clearly lower AUC (0.639 vs 0.652) -- noise-level given ~100 turns, not a real win. RandomForest was the best of all five models on Hindi (AUC 0.691 vs runner-up CatBoost 0.682), and every model hit the same 850ms floor there regardless of AUC, reconfirming the data-floor finding. No change to train.py; keeping RandomForest.

english (unchanged from RUN 7):
  AUC                 : 0.652
  mean response delay : 1120 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.45, delay=1000 ms

hindi (unchanged from RUN 7):
  AUC                 : 0.691
  mean response delay : 850 ms   <-- your score, lower is better
  interrupted turns   : 5.0%
  operating point     : threshold=0.05, delay=850 ms

## FINAL: shipped predict.py

changes -
train.py's per-run numbers above are all honest out-of-fold (cross-validated) estimates -- that's what should be trusted as the generalization estimate. For the actual deliverable, added fit_final_models.py (refits the same recipe -- window=2.5s, RandomForest n_estimators=100/max_depth=3/min_samples_leaf=5 -- on 100% of each language's labeled pauses, standard "validate with CV, ship trained-on-everything" practice) and predict.py (loads the saved model by data_dir folder name, falls back to a pooled model for an unrecognized folder, never retrains). Regenerated predictions_english.csv / predictions_hindi.csv via predict.py as the required deliverables; kept predictions_english_oof.csv / predictions_hindi_oof.csv as the honest reference copies since predict.py's model is fit on 100% of that folder's data, so scoring it against the SAME folder is in-sample and looks artificially better (AUC 0.848/0.851, delay 896ms/767ms) than true generalization -- do not read those as the real score. The RUN 7/8 numbers above (AUC 0.652/0.691, 1120ms/850ms) remain the credible estimate of how predict.py will perform on genuinely unseen pauses.



