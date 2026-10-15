"""Lesson 6: is model B really better than model A?

Two logistic regressions, one on feature block A and one on block B (`truth.Views`). When both blocks carry the
same signal, the two models are equally good by symmetry, so every "significant" difference is a false positive.
Five common tests at the 5% level, each run on many independent datasets:

  paired t-test on 10-fold CV       per-fold error differences, df 9
  resampled t-test                  15 random 2/3 - 1/3 splits, df 14 (the one Dietterich 1998 warned about)
  corrected resampled t-test        same splits, variance x (1/J + n_test/n_train) (Nadeau and Bengio 2003)
  5x2cv paired t-test               Dietterich 1998
  McNemar on one holdout            exact binomial test on the cases where exactly one model is right

The false-positive rate should be at most 5%. Then block B gets more signal than A, and the same tests' power
(share of datasets where B is correctly found better) is measured.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold

from . import config, truth

A, B = slice(0, 5), slice(5, 10)
ALPHA = 0.05


def diff(X, y, a, b):
    """Error of model A minus error of model B, trained on rows a and tested on rows b (positive = B better)."""
    ea = (LogisticRegression().fit(X[a][:, A], y[a]).predict(X[b][:, A]) != y[b]).mean()
    eb = (LogisticRegression().fit(X[a][:, B], y[a]).predict(X[b][:, B]) != y[b]).mean()
    return float(ea - eb)


def t_p(d, var_scale=None):
    d = np.asarray(d)
    v = d.var(ddof=1) * (var_scale if var_scale is not None else 1 / len(d))
    if v == 0:
        return 1.0
    return float(2 * stats.t.sf(abs(d.mean() / np.sqrt(v)), len(d) - 1))


def tests(X, y, seed):
    n = len(y)
    out = {}
    d = [diff(X, y, a, b) for a, b in KFold(10, shuffle=True, random_state=seed).split(X)]
    out["paired t-test on 10-fold CV"] = (t_p(d), np.mean(d))
    ss = list(ShuffleSplit(15, test_size=1 / 3, random_state=seed).split(X))
    d = [diff(X, y, a, b) for a, b in ss]
    out["resampled t-test (15 splits)"] = (t_p(d), np.mean(d))
    n_te = len(ss[0][1])
    out["corrected resampled t-test"] = (t_p(d, 1 / len(d) + n_te / (n - n_te)), np.mean(d))
    p, s2 = [], []
    for i in range(5):
        f = list(StratifiedKFold(2, shuffle=True, random_state=seed * 10 + i).split(X, y))
        p1, p2 = diff(X, y, *f[0]), diff(X, y, *f[1])
        p.append(p1)
        s2.append((p1 - (p1 + p2) / 2) ** 2 + (p2 - (p1 + p2) / 2) ** 2)
    t = p[0] / np.sqrt(np.mean(s2)) if np.mean(s2) > 0 else 0.0
    out["5x2cv paired t-test"] = (float(2 * stats.t.sf(abs(t), 5)), p[0])
    a, b = ss[0]
    ra = LogisticRegression().fit(X[a][:, A], y[a]).predict(X[b][:, A]) == y[b]
    rb = LogisticRegression().fit(X[a][:, B], y[a]).predict(X[b][:, B]) == y[b]
    k01, k10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    out["McNemar on one holdout"] = (stats.binomtest(k01, k01 + k10, 0.5).pvalue if k01 + k10 else 1.0,
                                     (k10 - k01) / len(b))
    return out


def one(args):
    seed, bsig, n = args
    X, y = truth.Views(a=0.35, b=bsig).sample(n, np.random.default_rng(seed))
    return {name: {"reject": p < ALPHA, "says B better": (p < ALPHA) and (dm > 0)}
            for name, (p, dm) in tests(X, y, seed).items()}


def true_gap(b_better, n, draws=30):
    """Average true error of each model over independent training sets of size n, on 200,000 fresh rows."""
    gen = truth.Views(b=b_better)
    Xp, yp = gen.sample(200_000, np.random.default_rng(truth.SEED))
    e = np.array([[(LogisticRegression().fit(X[:, blk], y).predict(Xp[:, blk]) != yp).mean() for blk in (A, B)]
                  for X, y in (gen.sample(n, np.random.default_rng(100 + s)) for s in range(draws))])
    return {"model A error": float(e[:, 0].mean()), "model B error": float(e[:, 1].mean()),
            "gap": float((e[:, 0] - e[:, 1]).mean())}


def run(sims=500, n=300, b_better=0.5, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for label, bsig in [("equal models (false-positive rate)", 0.35), ("B truly better (power)", b_better)]:
        with ProcessPoolExecutor(workers) as ex:
            r = list(ex.map(one, [(s, bsig, n) for s in range(sims)], chunksize=10))
        key = "reject" if bsig == 0.35 else "says B better"
        out[label] = {name: float(np.mean([x[name][key] for x in r])) for name in r[0]}
    res = {"sims": sims, "n": n, "alpha": ALPHA, "rates": out, "true errors when B is better": true_gap(b_better, n)}
    (config.RESULTS / "l6_compare.json").write_text(json.dumps(res, indent=1))
    print(pd.DataFrame(out).round(3))
    plot(out)
    return res


def plot(out):
    import matplotlib.pyplot as plt
    d = pd.DataFrame(out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, col, ref in zip(axes, d.columns, [ALPHA, None]):
        ax.barh(y, d[col], color="#2a6fbb")
        if ref:
            ax.axvline(ref, color="#b8422e", lw=1.5)
            ax.text(ref + 0.005, y.max() + 0.35, "5% promised", color="#b8422e", fontsize=8)
        ax.set_yticks(y, d.index, fontsize=9)
        ax.set_title(col if ref else col + "\nonly meaningful for tests that keep their 5% promise", fontsize=10)
        ax.set_xlabel("share of datasets")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "l6_compare.png", dpi=130)
    plt.close(fig)
