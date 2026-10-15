# TODO

## Lessons
- [ ] 1 Classification metrics: imbalance (accuracy vs balanced accuracy), ROC vs PR AUC at 1%/10%/50% prevalence,
      log loss and Brier, reliability diagram, threshold from a cost ratio. Real data: a credit or fraud set.
- [ ] 2 Regression: MAE (median-optimal) vs RMSE (mean-optimal) vs MAPE (breaks near zero, favours under-prediction)
      vs R^2; pinball loss and prediction-interval coverage; residual plots. Real data: California housing.
- [x] 3 Estimating test error (results below)
- [x] 4 Leakage (results below)
- [ ] 4c follow-up: purging made little difference with 5 folds and h = 20; show when it matters (more folds,
      longer horizons)
- [ ] 5 Tuning: non-nested vs nested CV optimism as the number of configurations grows; winner's curse;
      choosing on a single validation group (the ast-specialists MIMII lesson: valve probe 0.96 val -> 0.61 test)
- [ ] 6 Comparing models: false-positive rate of the naive paired t-test on CV folds vs the corrected one, by simulation
- [ ] 7 Learning curves: high-bias vs high-variance signatures; when more data helps
- [ ] 8 Model-quality report function (confusion matrix, ROC, PR, calibration, permutation importance); AutoML
      baseline optional (FLAML, CPU)

## Docs
- [ ] README results for each lesson as it lands
- [ ] Executed walkthrough notebook (`notebooks/`)
- [ ] Blog draft
- [ ] Push to GitHub (private first)
