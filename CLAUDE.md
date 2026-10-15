# eval-lab

Trello #20 (Learn about Evaluating ML Models). **Question: how much should you trust a model's score?**
Every lesson uses data from a generator whose truth is known (`truth.py`), so each evaluation shortcut is
measured against the real out-of-sample score instead of argued about.

## Run

```bash
source ../../.venv/bin/activate          # shared workspace venv (sklearn 1.9, numpy 2.4); CPU only
pytest -q
PYTHONPATH=src python -m evallab estimators   # lesson 3, ~4 min on 8 processes
PYTHONPATH=src python -m evallab leakage      # lesson 4, ~40 s
```

Results: `reports/results/*.json` (+ per-simulation CSVs), figures in `reports/figures/`.

## Lessons (status in TODO.md)

1. Choosing the classification metric: imbalance, ROC vs PR AUC, proper scores (log loss, Brier), calibration,
   cost-based thresholds.
2. Regression evaluation: MAE vs RMSE vs MAPE vs R^2, what each one's best prediction is, intervals and coverage.
3. **Estimating test error** (done): holdout, K-fold, repeated K-fold, .632 bootstrap vs the true error.
4. **Leakage** (done): selection before CV, grouped records, overlapping time targets.
5. Tuning and selection bias: nested CV, the winner's curse, choosing on one validation group.
6. Comparing two models: paired bootstrap, McNemar, corrected resampled t-test vs the naive t-test.
7. Learning curves and bias/variance diagnosis.
8. A model-quality report (SageMaker Autopilot style) for any sklearn model; optionally an AutoML baseline.

## Key decisions

- **Truth by simulation, then one real dataset per lesson.** Simulation measures the error of an evaluation
  method; the real dataset shows the same effect where truth is unknown.
- Seeds are fixed (`truth.SEED`, per-simulation seeds = simulation index). Truth uses 100,000 fresh draws.
- Simulations run in a process pool; each is independent.
- Lesson 3 scores misclassification error to match ESL 7.12; lesson 1 explains why a proper score is usually
  the better choice.
- References are in PRIOR_ART.md, found through textbook-kb.
