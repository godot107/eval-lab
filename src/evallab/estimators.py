"""Lesson 3: how good is each estimate of test error?

One problem (`truth.Logistic`), many independent training sets. For each training set T:
  truth     Err_T: the fitted model's error on 100,000 fresh draws (what we'd like to know)
  estimates holdout 70/30, 5-fold CV, 10-fold CV, repeated 5x5-fold CV, .632 bootstrap, and training error

and then, across training sets, how far each estimate is from its own Err_T: bias, spread (sd of the error), and
whether it even moves with Err_T (correlation). ESL 7.12 predicts the last one is near zero: cross-validation
estimates the *average* error of the learning procedure, not the error of the particular model you trained.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split

from . import config, truth


def err(model, X, y):
    return float((model.predict(X) != y).mean())


def cv_err(model, X, y, splitter):
    e = [err(clone(model).fit(X[a], y[a]), X[b], y[b]) for a, b in splitter.split(X)]
    return float(np.mean(e))


def boot632(model, X, y, rng, B=50):
    """Efron's .632 estimator: 0.368 x training error + 0.632 x out-of-bag error (ESL 7.11)."""
    n, oob = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        o = np.setdiff1d(np.arange(n), i)
        if len(np.unique(y[i])) < 2 or len(o) == 0:
            continue
        oob.append(err(clone(model).fit(X[i], y[i]), X[o], y[o]))
    return 0.368 * err(clone(model).fit(X, y), X, y) + 0.632 * float(np.mean(oob))


def one(seed, gen, n, model, Xpop, ypop):
    rng = np.random.default_rng(seed)
    X, y = gen.sample(n, rng)
    fit = clone(model).fit(X, y)
    a, b, ya, yb = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    return {"truth": err(fit, Xpop, ypop),
            "training error": err(fit, X, y),
            "holdout 70/30": err(clone(model).fit(a, ya), b, yb),
            "5-fold CV": cv_err(model, X, y, KFold(5, shuffle=True, random_state=seed)),
            "10-fold CV": cv_err(model, X, y, KFold(10, shuffle=True, random_state=seed)),
            "5x5-fold CV": cv_err(model, X, y, RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)),
            ".632 bootstrap": boot632(model, X, y, rng)}


def run(sims=200, n=200, workers=8):
    gen = truth.Logistic()
    Xpop, ypop = gen.sample(truth.POP, np.random.default_rng(truth.SEED))
    model = LogisticRegression(C=1.0, max_iter=1000)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:                 # each training set is independent
        rows = list(ex.map(one, range(sims), [gen] * sims, [n] * sims, [model] * sims,
                           [Xpop] * sims, [ypop] * sims, chunksize=8))
    d = pd.DataFrame(rows)
    out = {"sims": sims, "n": n, "p": gen.p, "bayes_error": gen.bayes_error(),
           "mean_truth": float(d.truth.mean()), "sd_truth": float(d.truth.std()), "methods": {}}
    for m in d.columns.drop("truth"):
        gap = d[m] - d.truth
        out["methods"][m] = {"mean": float(d[m].mean()), "bias": float(gap.mean()), "sd_of_gap": float(gap.std()),
                             "rmse": float(np.sqrt((gap ** 2).mean())), "corr_with_truth": float(d[m].corr(d.truth))}
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "l3_estimators.json").write_text(json.dumps(out, indent=1))
    d.to_csv(config.RESULTS / "l3_estimators.csv", index=False)
    plot(d, out)
    return out


def plot(d, out):
    import matplotlib.pyplot as plt
    ms = [m for m in out["methods"] if m != "training error"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    gaps = [d[m] - d.truth for m in ms]
    axes[0].boxplot(gaps, vert=False, tick_labels=ms)
    axes[0].axvline(0, color="0.4", lw=1)
    axes[0].set_xlabel("estimate - true error of that model")
    axes[0].set_title("How far off is each estimate?", fontsize=10)
    axes[1].scatter(d.truth, d["10-fold CV"], s=10, alpha=0.6)
    lo, hi = d[["truth", "10-fold CV"]].min().min(), d[["truth", "10-fold CV"]].max().max()
    axes[1].plot([lo, hi], [lo, hi], color="0.4", lw=1)
    axes[1].set_xlabel("true error of the fitted model")
    axes[1].set_ylabel("10-fold CV estimate")
    axes[1].set_title(f"Does CV track the model it scores? corr = {out['methods']['10-fold CV']['corr_with_truth']:.2f}",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "l3_estimators.png", dpi=130)
    plt.close(fig)
