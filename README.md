# eval-lab: how much should you trust a model's score?

Every model comes with a number: 92% accuracy, AUC 0.85, R² 0.3. This lab asks how wrong that number can be,
and **measures** it. Each lesson uses data from a generator whose truth is known, so an evaluation method's
estimate can be compared with the model's real score on 100,000 fresh examples. Most tutorials show how to call
`cross_val_score`; this one shows how far off it can be, and when.

## Results so far

### Lesson 3: a cross-validation score is a noisy estimate

A logistic regression on 200 examples, repeated over 200 independent training sets. The best possible error is
23.9%; the fitted models' true error averaged 27.0%.

| Estimate | Average gap from the truth | Typical size of the gap (sd) | Correlation with the truth |
|---|---|---|---|
| Training error | **-6.6 points** (optimistic) | 3.3 | 0.15 |
| Holdout 70/30 | +1.0 | **6.1** | 0.07 |
| 5-fold CV | +0.7 | 3.6 | 0.17 |
| 10-fold CV | +0.3 | 3.6 | 0.18 |
| 5x5 repeated CV | +0.9 | 3.4 | 0.13 |
| .632 bootstrap | -0.8 | 3.2 | 0.15 |

![Lesson 3](reports/figures/l3_estimators.png)

- **Training error flatters**: 6.6 points too good on average.
- **A single holdout split is the noisiest**: its gap is about twice as wide as K-fold's.
- **K-fold is nearly unbiased but still noisy**: the estimate is typically 3.6 points off, while the true error
  of the fitted models only varied by 0.9 points between training sets. So a CV score of 27% says "somewhere around
  20-34%", and **it barely tracks whether this particular model is better or worse than usual** (correlation
  0.18). That is ESL's point in section 7.12: CV estimates how good the *procedure* is on average, not the one
  model you trained. Repeating the CV doesn't fix that (the noise comes from the 200 examples, not the split).

### Lesson 4: leakage, measured

![Lesson 4](reports/figures/l4_leakage.png)

| Leak | What the evaluation said | Truth |
|---|---|---|
| Choosing 100 of 5,000 features using all the labels, *then* cross-validating (ESL 7.10.2) | 1.3% error | **50%** (the features are pure noise); 48.7% when selection is done inside each fold |
| Ten records per patient, random K-fold | AUC 0.73 | **0.68** on new patients; group K-fold says 0.66 |
| Overlapping 20-day targets on a pure-noise series, shuffled K-fold | R² **+0.02** (a small edge) | **-0.20** (it loses); contiguous or purged folds say -0.29 |

The time-series case is the dangerous one: shuffled CV didn't just inflate the score, it turned a losing model
into one that looks like it has an edge. Purging and embargo (Lopez de Prado ch. 7) changed little over plain
contiguous folds here, because with 5 folds only the rows at each fold boundary leak; it matters more with more
folds and longer horizons (on the TODO list).

## Lessons

1. Choosing the classification metric (planned)
2. Evaluating regression (planned)
3. Estimating test error (done)
4. Leakage (done)
5. Tuning and selection bias: nested CV (planned)
6. Comparing two models fairly (planned)
7. Learning curves (planned)
8. A model-quality report for any model, AutoML-style (planned)

## Run

```bash
pip install -r requirements-dev.txt
pytest -q
PYTHONPATH=src python -m evallab estimators
PYTHONPATH=src python -m evallab leakage
```

References and what each lesson takes from them: [PRIOR_ART.md](PRIOR_ART.md).
