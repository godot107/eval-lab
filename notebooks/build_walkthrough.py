"""Build notebooks/walkthrough.ipynb, then execute it:

    PYTHONPATH=src python notebooks/build_walkthrough.py
    PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

Small live demos import evallab; the full simulations are read from reports/results (`python -m evallab all`).
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def fig(name):
    return code(f'display(Image(str(config.FIGURES / "{name}")))')


cells = [
    md("""# How much should you trust a model's score?

Every model ships with a number: 92% accuracy, AUC 0.85, R² 0.3. This notebook shows how far that number can be
from the model's real performance, and which habits make it worse.

**The trick:** all the data here comes from simulations where the truth is known. So besides the usual evaluation,
every model is also scored on 100,000 fresh examples it has never seen. That fresh-data score is the truth the
evaluation is trying to estimate, and the gap between the two is what each lesson measures.

| Lesson | Question |
|---|---|
| 1 | Which classification metric, and which threshold? |
| 2 | Which regression metric, and can I trust an interval? |
| 3 | How noisy is a cross-validation score? |
| 4 | How does information leak from test into training? |
| 5 | Does tuning make the score optimistic? |
| 6 | Is model B really better than model A? |
| 7 | Will more data help? |
| 8 | Putting it together: a model-quality report |"""),
    code("""import json
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from evallab import config, leakage, regression, truth

R = config.RESULTS
load = lambda name: json.loads((R / name).read_text())
pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)"""),

    md("""## Lesson 1: choosing the classification metric

### 1a. Rare positives

The same kind of problem with 50%, 10% and 1% positives. "Always 'no'" is a model that never predicts the rare
class."""),
    code("""l1 = load("l1_classification.json")
pd.DataFrame(l1["1a_imbalance"]).T[["accuracy", "balanced accuracy", "ROC AUC", "PR AUC", "Brier skill"]]"""),
    md("""At 1% positives the model's accuracy equals the do-nothing model's (99%). ROC AUC hardly moves as positives
get rarer, because it only asks whether positives rank above negatives. PR AUC falls toward the prevalence, because
it asks how many of the flagged cases are real, and that is what gets hard when positives are rare. For rare
events, report PR AUC (or precision at a useful recall) next to ROC AUC, and never accuracy alone.

### 1b. Are the probabilities honest?"""),
    code("""pd.DataFrame(l1["1b_calibration"]).T[["ROC AUC", "log loss", "Brier", "calibration error", "mean prediction",
                                       "prevalence"]]"""),
    md("""`class_weight="balanced"` is a common fix for imbalance. It leaves ROC AUC unchanged (0.861 vs 0.862), so a
ranking metric can't see anything wrong, but every probability is inflated: on average it predicts 34% when the
real rate is 10%. Log loss and Brier score (proper scoring rules: they reward honest probabilities) and the
calibration error catch it. Recalibrating on a held-out split fixes it.

### 1c. Choosing the threshold from costs

Missing a positive costs 5; a false alarm costs 1. With honest probabilities, flag a case when its probability is
above 1/(1+5) = 0.167."""),
    code("""pd.DataFrame(l1["1c_thresholds"]).T"""),
    fig("l1_classification.png"),
    md("""The cost-based threshold gets within 0.006 of the best possible cost. The default 0.5 costs 37% more. And
the same cost threshold applied to the *balanced* model is the worst option of all, because its probabilities are
not probabilities. Thresholds and calibration go together.

### 1d. Real data: German credit

1,000 loans, 30% bad. Approving a bad loan costs 5; rejecting a good one costs 1."""),
    code("""print(l1["1d_info"])
pd.DataFrame(l1["1d_german"]).T"""),
    md("""At the default 0.5 threshold, both models cost more per loan than simply rejecting everyone (0.70). At the
cost threshold the logistic model costs 0.54. The model didn't change; the decision rule did."""),

    md("""## Lesson 2: evaluating regression

### 2a. The metric decides the winner

A skewed target (like prices). Five models, each built for one metric, scored on 100,000 fresh rows. Rows starting
with `oracle` use the true distribution: the best possible prediction for each metric."""),
    code("""l2 = load("l2_regression.json")
pd.DataFrame(l2["2a_metric_choice"]).T[["RMSE", "MAE", "MAPE", "pinball 0.9", "mean bias"]]"""),
    md("""MAE rewards the median, MAPE rewards a value *below* the median (it pays you to forecast low), pinball loss
rewards the chosen quantile. Only the mean model is unbiased, which matters when forecasts are summed.

### 2b. R² depends on the data, not just the model (live)"""),
    code("""pd.DataFrame(regression.r2_range(n=50_000))"""),
    md("""Same model, same noise, same error. R² differs only because the inputs are more spread out in one sample.

### 2c. Does an 80% interval contain 80%?"""),
    code("""pd.DataFrame(l2["2c_intervals"]).T"""),
    fig("l2_regression.png"),
    md("""Overall coverage can be right while the interval is far too wide where things are calm and too narrow
where risk is highest. Conformalized quantile regression (CQR) gets closest to the true interval.

### 2d. Real data: California housing"""),
    code("""display(pd.DataFrame(l2["2d_california"]).T[["RMSE", "MAE", "MAPE", "pinball 0.9", "R2"]])
cap = l2["2d_cap"]
print(f"{cap['capped_rows']} of {cap['n']} homes are recorded at the $500k cap (true price unknown, higher)")"""),

    md("""## Lesson 3: a cross-validation score is a noisy estimate

A logistic regression on 200 examples, 200 independent training sets. For each, the true error of the fitted
model is compared with each estimate of it."""),
    code("""l3 = load("l3_estimators.json")
print(f"best possible error {l3['bayes_error']:.3f}; fitted models' true error {l3['mean_truth']:.3f} "
      f"(varies by {l3['sd_truth']:.3f} between training sets)")
pd.DataFrame(l3["methods"]).T[["bias", "sd_of_gap", "corr_with_truth"]]"""),
    fig("l3_estimators.png"),
    md("""K-fold CV is right on average but typically 3.6 points off for any one training set, and it hardly moves with
the error of the particular model you trained (right panel; Hastie et al., *Elements of Statistical Learning*,
7.12). A single holdout is twice as noisy. Training error flatters by 6.6 points."""),

    md("""## Lesson 4: leakage

### 4a. The wrong way to cross-validate (live)

Fifty examples, 5,000 random features, labels that are coin flips: the true error is 50%. Choose the 100 features
most related to the label using *all* the labels, then cross-validate:"""),
    code("""pd.DataFrame([leakage.wrong_vs_right(seed) for seed in range(10)]).describe().loc[["mean", "min", "max"]]"""),
    md("""### 4b and 4c. Same patient in both folds; overlapping time targets"""),
    code("""l4 = load("l4_leakage.json")
pd.concat({k: pd.DataFrame(v["results"]).T[["mean", "p10", "p90"]] for k, v in l4.items()})"""),
    fig("l4_leakage.png"),
    md("""On a time series with nothing to predict, shuffled CV reports an edge; the model actually loses. When does
purging (dropping training rows whose targets overlap the test period) matter? More folds and longer horizons:"""),
    code("""pd.DataFrame(load("l4c_purge_sweep.json"))"""),

    md("""## Lesson 5: tuning makes the winning score optimistic

Try m configurations, keep the one with the best cross-validation score, and report that score: it's the winner of
m noisy estimates. Nested CV puts the whole tuning procedure inside an outer loop."""),
    code("""l5 = load("l5_tuning.json")
pd.DataFrame(l5["5a_nested"])"""),
    md("""### Choosing on a single validation group

Groups (machines, sites) each bend the shared pattern. Choose a configuration using one held-out group, or
leave-one-group-out CV over all twelve; then score the choice on 300 new groups."""),
    code("""pd.concat({k: pd.DataFrame(v) for k, v in l5["5b_one_group"].items()})"""),
    fig("l5_tuning.png"),

    md("""## Lesson 6: is model B really better?

Two models that are exactly equally good (by construction). Each test is run on 500 independent datasets; at the 5%
level, at most 5% should call a winner. Then B is made truly better and the same tests' power is measured."""),
    code("""l6 = load("l6_compare.json")
display(pd.DataFrame(l6["rates"]))
l6["true errors when B is better"]"""),
    fig("l6_compare.png"),
    md("""The resampled t-test on repeated random splits calls equal models different almost half the time: the
splits share most of their training data, so their results aren't independent and the test's variance is far too
small (Dietterich 1998). The correction of Nadeau and Bengio (2003), the 5x2cv test and McNemar's test keep their
promise. Power is low for all honest tests at 300 rows, even with a 6-point true difference: small datasets can't
separate close models."""),

    md("""## Lesson 7: will more data help?

A curved class boundary; three models. Training, cross-validation and true error by training size; the red line is
the best possible error."""),
    code("""l7 = load("l7_curves.json")
lc = pd.DataFrame(l7["learning"])
display(lc.pivot(index="n", columns="model", values="truth"))
print("tree depth chosen by CV:", l7["depth chosen by CV"], "| best depth on fresh data:", l7["depth best on truth"])"""),
    fig("l7_curves.png"),
    md("""- **High bias** (logistic regression): training and CV error meet, far above the floor. More data won't help;
  a more flexible model will.
- **High variance** (1-nearest neighbour): training error 0, a large gap to CV. More data helps, slowly.
- **Validation curve**: CV error tracks the truth closely across tree depths and picks the right depth (4); the
  training error keeps falling and would pick the deepest tree."""),

    md("""## Lesson 8: a model-quality report

`report.model_report` combines the lessons: an untouched holdout set aside first (lesson 5), a leaderboard with a
no-skill baseline, proper scores, calibration and cost with bootstrap intervals (lessons 1 and 3), group-aware folds
when there are groups (lesson 4), and a holdout check with confusion matrix, curves, permutation importance and a
learning curve (lesson 7). Here on German credit:"""),
    code("""display(Markdown((config.REPORTS / "german_credit_report.md").read_text().replace("figures/", "../reports/figures/")))"""),
    fig("l8_german_credit_report.png"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("walkthrough.ipynb")
nbf.write(nb, out)
print(out)
