"""Lesson 4: leakage, three ways. In each, the true score is known, so the leak is measured, not argued.

4a  The wrong way to cross-validate (ESL 7.10.2). 50 samples, 5,000 noise features, labels independent of all
    of them, so the true error is 50%. Pick the 100 features most correlated with the label using *all* the
    data, then cross-validate a 1-nearest-neighbour classifier on them: the CV error comes out far below 50%.
    The right way puts the selection inside each fold (a Pipeline).

4b  Grouped records. Ten records per patient; a patient's records share a feature signature and a hidden risk.
    Random K-fold puts the same patient in training and test, so the model 'recognises' the patient.
    GroupKFold keeps patients apart. Truth: the model scored on brand-new patients.

4c  Overlapping time-series targets. The series is pure noise (true R^2 <= 0), but each target sums the next h
    days and features are long trailing means, so neighbouring rows are near-copies. Shuffled K-fold, plain
    (contiguous) K-fold, purged + embargoed K-fold (Lopez de Prado ch. 7) and walk-forward are compared.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from . import config, truth


# ---------- 4a: selection outside the folds ----------

def wrong_vs_right(seed, n=50, p=5000, k=100):
    rng = np.random.default_rng(seed)
    X, y = truth.noise_wide(n, p, rng)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    knn = KNeighborsClassifier(1)
    keep = SelectKBest(f_classif, k=k).fit(X, y).get_support()          # sees every label, test folds included
    wrong = 1 - cross_val_score(knn, X[:, keep], y, cv=cv).mean()
    right = 1 - cross_val_score(make_pipeline(SelectKBest(f_classif, k=k), knn), X, y, cv=cv).mean()
    return {"wrong way (select, then CV)": float(wrong), "right way (select inside each fold)": float(right)}


# ---------- 4b: groups ----------

def grouped(seed, n_groups=60, new_groups=2000):
    gen = truth.Grouped()
    rng = np.random.default_rng(seed)
    X, y, g = gen.sample(n_groups, rng)
    Xn, yn, _ = gen.sample(new_groups, rng)
    model = ExtraTreesClassifier(300, min_samples_leaf=2, n_jobs=1, random_state=seed)

    def cv_auc(splitter, groups=None):
        s = np.zeros(len(y))
        for a, b in splitter.split(X, y, groups):
            s[b] = clone(model).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        return float(roc_auc_score(y, s))

    fit = clone(model).fit(X, y)
    return {"random K-fold": cv_auc(KFold(5, shuffle=True, random_state=seed)),
            "group K-fold": cv_auc(GroupKFold(5), g),
            "truth (new patients)": float(roc_auc_score(yn, fit.predict_proba(Xn)[:, 1]))}


# ---------- 4c: time ----------

def purged_folds(t, h, k=5, embargo=60):
    """Contiguous test blocks. Drop training rows whose target window overlaps the test block's (purge), and
    `embargo` more rows after it (here the longest feature window), since trailing features carry the test
    period forward."""
    e = embargo
    for b in np.array_split(np.arange(len(t)), k):
        lo, hi = t[b[0]], t[b[-1]]
        keep = (t + h <= lo) | (t > hi + h + e)
        yield np.flatnonzero(keep), b


def walk_forward(t, h, k=5):
    """Train on everything before the test block (minus h rows of target overlap); test on the block."""
    blocks = np.array_split(np.arange(len(t)), k + 1)
    for b in blocks[1:]:
        yield np.flatnonzero(t + h <= t[b[0]]), b


def timeseries(seed, n=2000, h=20):
    rng = np.random.default_rng(seed)
    X, y, t = truth.overlapping(n, h, rng)
    Xn, yn, _ = truth.overlapping(20_000, h, rng)
    model = KNeighborsRegressor(10)

    def cv_r2(folds):
        r = [r2_score(y[b], clone(model).fit(X[a], y[a]).predict(X[b])) for a, b in folds if len(a) > 50]
        return float(np.mean(r))

    return {"shuffled K-fold": cv_r2(KFold(5, shuffle=True, random_state=seed).split(X)),
            "contiguous K-fold": cv_r2(KFold(5).split(X)),
            "purged + embargo K-fold": cv_r2(purged_folds(t, h)),
            "walk-forward": cv_r2(walk_forward(t, h)),
            "truth (fresh series)": float(r2_score(yn, clone(model).fit(X, y).predict(Xn)))}


# ---------- run all ----------

def _many(fn, sims, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(fn, range(sims))))


def summary(d):
    return {c: {"mean": float(d[c].mean()), "p10": float(d[c].quantile(0.1)), "p90": float(d[c].quantile(0.9))}
            for c in d.columns}


def run(sims=50, workers=8):
    out = {}
    for name, fn, unit in [("4a_wrong_way", wrong_vs_right, "error (truth 0.50)"),
                           ("4b_grouped", grouped, "ROC AUC"),
                           ("4c_time", timeseries, "R^2 (truth <= 0)")]:
        d = _many(fn, sims, workers)
        out[name] = {"unit": unit, "sims": sims, "results": summary(d)}
        d.to_csv(config.RESULTS / f"l{name}.csv", index=False)
        print(name, json.dumps(out[name]["results"], indent=1))
    (config.RESULTS / "l4_leakage.json").write_text(json.dumps(out, indent=1))
    plot(out)
    return out


def plot(out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (name, r) in zip(axes, out.items()):
        res = r["results"]
        labels = list(res)
        m = [res[k]["mean"] for k in labels]
        lo = [res[k]["mean"] - res[k]["p10"] for k in labels]
        hi = [res[k]["p90"] - res[k]["mean"] for k in labels]
        y = np.arange(len(labels))[::-1]
        colors = ["#b8422e" if "truth" in k else "#2a6fbb" for k in labels]
        ax.errorbar(m, y, xerr=[lo, hi], fmt="none", ecolor="0.6", capsize=3)
        ax.scatter(m, y, c=colors, zorder=3)
        ax.set_yticks(y, labels)
        ax.set_xlabel(r["unit"])
        ax.set_title({"4a_wrong_way": "4a Feature selection before CV", "4b_grouped": "4b Same patient in both folds",
                      "4c_time": "4c Overlapping time targets"}[name], fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        if name == "4a_wrong_way":
            ax.axvline(0.5, color="#b8422e", lw=1.5)
            ax.text(0.49, y.mean(), "truth 0.50", color="#b8422e", ha="right", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "l4_leakage.png", dpi=130)
    plt.close(fig)
