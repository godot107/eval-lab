"""Lesson 2: evaluating regression.

2a  Each metric rewards a different prediction. On a skewed target (`truth.Skewed`), five models are trained,
    each aiming at one metric's best prediction (mean, median, a weighted median for MAPE, the 90% quantile, and
    the common "fit log y, then exponentiate"), and every model is scored by every metric on 100,000 fresh rows.
    The truth-optimal ("oracle") predictions are scored too. Each metric crowns the model built for it, so
    "which model is best?" has no answer until the metric is chosen, and the metric is a business decision:
    squared error punishes big misses, absolute error treats every unit alike, MAPE treats every percent alike
    (and quietly rewards under-forecasting).

2b  R^2 is not a property of the model. Same model, same noise, same RMSE; only the spread of x differs. R^2 changes
    from about 0.57 to 0.92, because it measures error relative to how much the target varies in *this* sample.

2c  Prediction intervals: does a "10%-90%" interval contain 80% of outcomes, and where? With noise that grows with
    x (`truth.Hetero`): a naive interval (prediction +/- 1.28 x RMSE), quantile regression, split conformal
    (constant width from calibration residuals) and conformalized quantile regression (CQR; Romano, Patterson and
    Candes 2019). Coverage is measured overall and in the quiet / middle / noisy thirds of x.

2d  Real data (California housing, 5-fold CV): the 2a table where truth is unknown, and the effect of the price cap.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from . import config, truth

Q = 0.9


def gbm(loss="squared_error", **kw):
    return HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05, random_state=truth.SEED, **kw)


class LogTarget:
    """Fit log y by least squares, predict exp(.): a common habit. It predicts the median, not the mean."""

    def __init__(self, base=None):
        self.base = base or gbm()

    def fit(self, X, y, sample_weight=None):
        self.m = clone(self.base).fit(X, np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.m.predict(X))


class Weighted:
    """Absolute-error model with weights 1/y: the weighted median, which is what minimises MAPE."""

    def __init__(self):
        self.m = gbm("absolute_error")

    def fit(self, X, y):
        self.m.fit(X, y, sample_weight=1 / y)
        return self

    def predict(self, X):
        return self.m.predict(X)


def mape(y, p):
    return float(np.mean(np.abs(y - p) / np.abs(y)))


METRICS = {"RMSE": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
           "MAE": lambda y, p: float(mean_absolute_error(y, p)),
           "MAPE": mape,
           "pinball 0.9": lambda y, p: float(mean_pinball_loss(y, p, alpha=Q)),
           "R2": lambda y, p: float(r2_score(y, p)),
           "mean bias": lambda y, p: float(np.mean(p - y))}


def models():
    return {"squared error (mean)": gbm(), "absolute error (median)": gbm("absolute_error"),
            "MAPE-weighted (low median)": Weighted(), "quantile 0.9": gbm("quantile", quantile=Q),
            "log target, exp back": LogTarget()}


def score_table(preds, y):
    return pd.DataFrame({name: {m: f(y, p) for m, f in METRICS.items()} for name, p in preds.items()}).T


def metric_choice(n=5000):
    gen = truth.Skewed()
    rng = np.random.default_rng(truth.SEED)
    X, y = gen.sample(n, rng)
    Xp, yp = gen.sample(truth.POP, rng)
    preds = {name: m.fit(X, y).predict(Xp) for name, m in models().items()}
    oracle = {"oracle: " + t: gen.best(Xp, t, Q) for t in ["mean", "median", "mape", "quantile"]}
    return score_table({**preds, **oracle}, yp)


def r2_range(n=100_000, seed=truth.SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for name, w in [("narrow x, U(-1, 1)", 1), ("wide x, U(-3, 3)", 3)]:
        x = rng.uniform(-w, w, n)
        y = 2 * x + rng.standard_normal(n)
        p = LinearRegression().fit(x[:, None], y).predict(x[:, None])
        out[name] = {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": float(r2_score(y, p))}
    return out


def intervals(n=4000, lo=0.1, hi=0.9):
    """Train / calibration split of n rows; coverage and width on 100,000 fresh rows."""
    from scipy.stats import norm
    gen = truth.Hetero()
    rng = np.random.default_rng(truth.SEED)
    X, y = gen.sample(n, rng)
    Xp, yp = gen.sample(truth.POP, rng)
    tr, ca = np.arange(n) < n // 2, np.arange(n) >= n // 2
    z = norm.ppf(hi)
    mean = gbm().fit(X[tr], y[tr])
    rmse = np.sqrt(np.mean((y[ca] - mean.predict(X[ca])) ** 2))
    qlo, qhi = gbm("quantile", quantile=lo).fit(X[tr], y[tr]), gbm("quantile", quantile=hi).fit(X[tr], y[tr])

    def conformal(scores, level=hi - lo):
        k = int(np.ceil((len(scores) + 1) * level))
        return np.sort(scores)[min(k, len(scores)) - 1]

    r = conformal(np.abs(y[ca] - mean.predict(X[ca])))
    e = conformal(np.maximum(qlo.predict(X[ca]) - y[ca], y[ca] - qhi.predict(X[ca])))
    mp, lp, hp = mean.predict(Xp), qlo.predict(Xp), qhi.predict(Xp)
    xp = Xp[:, 0]
    true_mid = 2 * np.sin(xp) + xp
    bands = {"naive: prediction +/- 1.28 RMSE": (mp - z * rmse, mp + z * rmse),
             "quantile regression (10%, 90%)": (lp, hp),
             "split conformal (constant width)": (mp - r, mp + r),
             "conformalized quantile (CQR)": (lp - e, hp + e),
             "oracle (true 10%-90%)": (true_mid - z * gen.scale(xp), true_mid + z * gen.scale(xp))}
    third = np.digitize(xp, [-1, 1])               # 0 quiet, 1 middle, 2 noisy
    out = {}
    for name, (a, b) in bands.items():
        inside = (yp >= a) & (yp <= b)
        out[name] = {"coverage": float(inside.mean()),
                     **{f"coverage, {t} third": float(inside[third == i].mean())
                        for i, t in enumerate(["quiet", "middle", "noisy"])},
                     "mean width": float(np.mean(b - a))}
    return out


def california(k=5):
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(data_home=str(config.ROOT / "data" / "sk"))
    X, y = d.data, d.target
    preds = {name: np.zeros(len(y)) for name in models()}
    for a, b in KFold(k, shuffle=True, random_state=truth.SEED).split(X):
        for name, m in models().items():
            preds[name][b] = m.fit(X[a], y[a]).predict(X[b])
    capped = y >= 5.0
    table = score_table(preds, y)
    cap = {name: {"mean error on capped rows": float(np.mean(p[capped] - y[capped])),
                  "MAE on capped rows": float(np.mean(np.abs(p[capped] - y[capped]))),
                  "MAE on the rest": float(np.mean(np.abs(p[~capped] - y[~capped])))} for name, p in preds.items()}
    return table, {"n": int(len(y)), "capped_rows": int(capped.sum()), "by_model": cap}


def run():
    a = metric_choice()
    b = r2_range()
    c = intervals()
    d, cap = california()
    out = {"2a_metric_choice": a.to_dict(orient="index"), "2b_r2_range": b, "2c_intervals": c,
           "2d_california": d.to_dict(orient="index"), "2d_cap": cap}
    (config.RESULTS / "l2_regression.json").write_text(json.dumps(out, indent=1))
    with pd.option_context("display.width", 160, "display.precision", 3):
        print(a, "\n", pd.DataFrame(b), "\n", pd.DataFrame(c).T, "\n", d, "\n", json.dumps(cap, indent=1))
    plot(a, c)
    return out


def plot(a, c):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ms = ["RMSE", "MAE", "MAPE", "pinball 0.9"]
    t = a.loc[[i for i in a.index if not i.startswith("oracle")], ms]
    rel = t / t.min()                                          # 1.0 = best model on that metric
    im = axes[0].imshow(rel.values, cmap="Blues_r", vmin=1, vmax=1.6, aspect="auto")
    axes[0].set_xticks(range(len(ms)), ms)
    axes[0].set_yticks(range(len(rel)), rel.index)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            axes[0].text(j, i, f"{rel.values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                         color="white" if rel.values[i, j] < 1.25 else "black")
    axes[0].set_title("Score / best score on each metric (1.00 = best)", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    names = [n for n in c]
    thirds = ["coverage, quiet third", "coverage, middle third", "coverage, noisy third"]
    x = np.arange(len(thirds))
    for i, n in enumerate(names):
        axes[1].plot(x, [c[n][t] for t in thirds], marker="o", label=n, lw=2 if "oracle" in n else 1.2,
                     color="0.3" if "oracle" in n else None, ls="--" if "oracle" in n else "-")
    axes[1].axhline(0.8, color="0.5", lw=0.8)
    axes[1].set_xticks(x, ["quiet x", "middle x", "noisy x"])
    axes[1].set_ylabel("share of outcomes inside the interval")
    axes[1].set_title("Is an '80%' interval 80% everywhere?", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "l2_regression.png", dpi=130)
    plt.close(fig)
