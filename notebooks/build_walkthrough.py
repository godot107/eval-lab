"""Build notebooks/walkthrough.ipynb, then execute it:

    PYTHONPATH=src python notebooks/build_walkthrough.py
    PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

Small live demos import evallab; the full simulations are read from reports/results (run `python -m evallab ...`).
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# How much should you trust a model's score?

Every model ships with a number. This notebook shows, with data whose truth is known, how far that number can be
from the model's real performance, and which habits make it worse. The data comes from simulations, so for every
model we also know its true score on 100,000 fresh examples."""),
    code("""import json
import numpy as np
import pandas as pd
from IPython.display import Image, display

from evallab import config, leakage, regression, truth

R = config.RESULTS
pd.set_option("display.precision", 3)"""),
    md("""## 1. The wrong way to cross-validate (live)

Fifty examples, 5,000 random features, and labels that are pure coin flips: no model can beat 50% error. If we
first keep the 100 features most related to the label *using all fifty labels*, then cross-validate, the score
looks excellent. The selection has already seen the test folds."""),
    code("""pd.DataFrame([leakage.wrong_vs_right(seed) for seed in range(10)]).describe().loc[["mean", "min", "max"]]"""),
    md("""The fix is to treat feature selection as part of the model and redo it inside every fold (a scikit-learn
`Pipeline`). Then the estimate is honest: about 50%."""),
    md("""## 2. Leakage in three forms (full simulation)"""),
    code("""l4 = json.loads((R / "l4_leakage.json").read_text())
pd.concat({k: pd.DataFrame(v["results"]).T[["mean", "p10", "p90"]] for k, v in l4.items()})"""),
    code("""display(Image(str(config.FIGURES / "l4_leakage.png")))"""),
    md("""- **Same patient in both folds**: random K-fold lets the model recognise patients it has already seen.
  Group K-fold matches what happens with new patients.
- **Overlapping time targets**: shuffled folds put near-copies of each test row into training. On a series with
  nothing to predict, shuffled CV reports a small edge; the model actually loses."""),
    md("""## 3. Even honest cross-validation is noisy"""),
    code("""l3 = json.loads((R / "l3_estimators.json").read_text())
print(f"best possible error {l3['bayes_error']:.3f}; true error of fitted models {l3['mean_truth']:.3f} "
      f"(sd {l3['sd_truth']:.3f} across training sets)")
pd.DataFrame(l3["methods"]).T[["bias", "sd_of_gap", "corr_with_truth"]]"""),
    code("""display(Image(str(config.FIGURES / "l3_estimators.png")))"""),
    md("""A 10-fold CV score is right on average but typically several points off for any one training set, and it
hardly moves with the error of the particular model you trained (right panel). Treat it as an estimate with an
error bar, and compare models on the same folds."""),
    md("""## 4. Regression: the metric decides the winner"""),
    code("""l2 = json.loads((R / "l2_regression.json").read_text())
t = pd.DataFrame(l2["2a_metric_choice"]).T
t[["RMSE", "MAE", "MAPE", "pinball 0.9", "mean bias"]]"""),
    md("""Rows starting with `oracle` use the true distribution: they are the best possible prediction for each
metric. Note the MAPE oracle sits *below* the median (negative bias): MAPE rewards forecasting low."""),
    md("""## 5. R² depends on the data, not just the model (live)"""),
    code("""pd.DataFrame(regression.r2_range(n=50_000))"""),
    md("""## 6. Does an 80% interval contain 80%?"""),
    code("""pd.DataFrame(l2["2c_intervals"]).T"""),
    code("""display(Image(str(config.FIGURES / "l2_regression.png")))"""),
    md("""Overall coverage can be right while the interval is far too wide where things are calm and too narrow where
risk is highest. Check coverage by segment. Conformalized quantile regression gets closest to the true interval."""),
    md("""## 7. Real data: California housing"""),
    code("""pd.DataFrame(l2["2d_california"]).T[["RMSE", "MAE", "MAPE", "pinball 0.9", "R2"]]"""),
    code("""cap = l2["2d_cap"]
print(f"{cap['capped_rows']} of {cap['n']} homes are recorded at the $500k cap (true price unknown, higher)")
pd.DataFrame(cap["by_model"]).T"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("walkthrough.ipynb")
nbf.write(nb, out)
print(out)
