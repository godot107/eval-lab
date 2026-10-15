"""Lesson 8: a model-quality report for any scikit-learn classifier, in the spirit of SageMaker Autopilot's.

`model_report(models, X, y, ...)` puts the lessons together:
  1. An untouched holdout (20%, stratified) is set aside first (lesson 5: the leaderboard winner's CV score is
     optimistic, so the final number comes from data no choice has seen).
  2. Leaderboard on the rest by 5-fold CV, out-of-fold predictions, including a no-skill baseline (lesson 1:
     always compare with the dumbest model). Proper scores (log loss, Brier), ranking (ROC and PR AUC),
     calibration error and the expected cost at the cost-based threshold (lesson 1), each with a 90% bootstrap
     interval (lesson 3: a CV score is an estimate). With `groups`, folds and the bootstrap respect them (lesson 4).
  3. The winner (lowest log loss) is refit and scored once on the holdout, with a confusion matrix at the cost
     threshold, ROC, precision-recall and calibration curves, permutation importance on the holdout, and a
     learning curve (lesson 7).
Writes reports/<name>_report.md and a figure.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, StratifiedKFold, learning_curve, train_test_split

from . import config, truth
from .classification import ece

METRICS = {"log loss": lambda y, p: log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
           "Brier": brier_score_loss, "ROC AUC": roc_auc_score, "PR AUC": average_precision_score,
           "calibration error": ece}


def expected_cost(y, p, c_fn, c_fp):
    pred = p > c_fp / (c_fp + c_fn)
    return float((c_fn * (~pred & (y == 1)) + c_fp * (pred & (y == 0))).mean())


def boot_ci(y, p, fn, groups=None, B=500, seed=truth.SEED):
    rng = np.random.default_rng(seed)
    g = np.asarray(groups) if groups is not None else np.arange(len(y))
    ids, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ids))]
    vals = []
    for _ in range(B):
        i = np.concatenate([members[j] for j in rng.integers(0, len(ids), len(ids))])
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], p[i]))
    return [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))]


def _rows(X, i):
    return X.iloc[i] if hasattr(X, "iloc") else X[i]


def model_report(name, models, X, y, groups=None, cost=(5.0, 1.0), k=5, holdout=0.2):
    y = np.asarray(y)
    c_fn, c_fp = cost
    idx = np.arange(len(y))
    if groups is None:
        dev, test = train_test_split(idx, test_size=holdout, stratify=y, random_state=truth.SEED)
        folds = list(StratifiedKFold(k, shuffle=True, random_state=truth.SEED).split(dev, y[dev]))
    else:
        groups = np.asarray(groups)
        ug = np.random.default_rng(truth.SEED).permutation(np.unique(groups))
        test_g = ug[: max(1, int(len(ug) * holdout))]
        test, dev = idx[np.isin(groups, test_g)], idx[~np.isin(groups, test_g)]
        folds = list(GroupKFold(k).split(dev, y[dev], groups[dev]))
    models = {"no skill (predict the base rate)": DummyClassifier(strategy="prior"), **models}
    board, oof = {}, {}
    for mname, m in models.items():
        p = np.zeros(len(dev))
        for a, b in folds:
            p[b] = clone(m).fit(_rows(X, dev[a]), y[dev[a]]).predict_proba(_rows(X, dev[b]))[:, 1]
        oof[mname] = p
        yd, gd = y[dev], (groups[dev] if groups is not None else None)
        row = {}
        for metric, fn in {**METRICS, "expected cost": lambda yy, pp: expected_cost(yy, pp, c_fn, c_fp)}.items():
            row[metric] = float(fn(yd, p))
            row[metric + " 90% CI"] = boot_ci(yd, p, fn, gd)
        board[mname] = row
    lb = pd.DataFrame(board).T
    winner = lb.drop(index="no skill (predict the base rate)")["log loss"].astype(float).idxmin()
    fit = clone(models[winner]).fit(_rows(X, dev), y[dev])
    pt = fit.predict_proba(_rows(X, test))[:, 1]
    yt = y[test]
    thr = c_fp / (c_fp + c_fn)
    final = {metric: float(fn(yt, pt)) for metric, fn in METRICS.items()}
    final["expected cost"] = expected_cost(yt, pt, c_fn, c_fp)
    final["cost, approve/accept everyone"] = float(c_fn * yt.mean())
    final["cost, reject/flag everyone"] = float(c_fp * (1 - yt.mean()))
    final["confusion at cost threshold"] = confusion_matrix(yt, pt > thr, labels=[0, 1]).tolist()
    pi = permutation_importance(fit, _rows(X, test), yt, scoring="neg_log_loss", n_repeats=20,
                                random_state=truth.SEED)
    cols = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(X.shape[1])]
    imp = pd.Series(pi.importances_mean, cols).sort_values(ascending=False)
    sizes, tr, va = learning_curve(clone(models[winner]), _rows(X, dev), y[dev], cv=folds,
                                   train_sizes=np.linspace(0.15, 1.0, 6), scoring="neg_log_loss")
    lc = {"rows": sizes.tolist(), "training log loss": (-tr.mean(1)).tolist(), "CV log loss": (-va.mean(1)).tolist()}
    out = {"name": name, "n": int(len(y)), "positives": float(y.mean()), "cost": {"miss": c_fn, "false alarm": c_fp},
           "threshold": thr, "leaderboard": lb.to_dict(orient="index"), "winner": winner,
           "holdout": {"n": int(len(test)), **final}, "permutation importance (log loss)": imp.head(10).to_dict(),
           "learning curve": lc}
    (config.RESULTS / f"l8_{name}_report.json").write_text(json.dumps(out, indent=1))
    plot(name, yt, pt, thr, imp, lc)
    write_md(out)
    return out


def plot(name, yt, pt, thr, imp, lc):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    cm = confusion_matrix(yt, pt > thr, labels=[0, 1])
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_xticks([0, 1], ["predicted 0", "predicted 1"])
    axes[0].set_yticks([0, 1], ["actual 0", "actual 1"])
    axes[0].set_title(f"Holdout confusion at threshold {thr:.3f}", fontsize=10)
    f, t, _ = roc_curve(yt, pt)
    axes[1].plot(f, t)
    axes[1].plot([0, 1], [0, 1], color="0.6", lw=0.8)
    axes[1].set_title(f"ROC (AUC {roc_auc_score(yt, pt):.3f})", fontsize=10)
    axes[1].set_xlabel("false-positive rate")
    axes[1].set_ylabel("true-positive rate")
    pr, rc, _ = precision_recall_curve(yt, pt)
    axes[2].plot(rc, pr)
    axes[2].axhline(yt.mean(), color="0.6", lw=0.8)
    axes[2].set_title(f"Precision-recall (AP {average_precision_score(yt, pt):.3f}; grey = no skill)", fontsize=10)
    axes[2].set_xlabel("recall")
    axes[2].set_ylabel("precision")
    frac, mean = calibration_curve(yt, pt, n_bins=8, strategy="quantile")
    axes[3].plot(mean, frac, "o-")
    axes[3].plot([0, 1], [0, 1], color="0.6", lw=0.8)
    axes[3].set_title("Calibration (holdout)", fontsize=10)
    axes[3].set_xlabel("predicted probability")
    axes[3].set_ylabel("observed rate")
    top = imp.head(8)[::-1]
    axes[4].barh(range(len(top)), top.values, color="#2a6fbb")
    axes[4].set_yticks(range(len(top)), top.index, fontsize=8)
    axes[4].set_title("Permutation importance (log-loss increase)", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"l8_{name}_report.png", dpi=120)
    plt.close(fig)


def write_md(o):
    lb = pd.DataFrame(o["leaderboard"]).T
    cols = ["log loss", "Brier", "ROC AUC", "PR AUC", "calibration error", "expected cost"]
    lines = [f"# Model-quality report: {o['name']}", "",
             f"{o['n']} rows, {o['positives']:.0%} positive. Cost of a miss {o['cost']['miss']:g}, of a false alarm "
             f"{o['cost']['false alarm']:g}, so the decision threshold is {o['threshold']:.3f}.", "",
             "## Leaderboard (5-fold CV on the development split; 90% bootstrap intervals)", "",
             "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m, r in lb.iterrows():
        cells = [f"{r[c]:.3f} ({r[c + ' 90% CI'][0]:.3f}-{r[c + ' 90% CI'][1]:.3f})" for c in cols]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    h = o["holdout"]
    lines += ["", f"## Winner on the untouched holdout ({h['n']} rows): {o['winner']}", "",
              "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {h[k]:.3f} |" for k in [*METRICS, "expected cost", "cost, approve/accept everyone", "cost, reject/flag everyone"]]
    tn, fp, fn, tp = np.ravel(h["confusion at cost threshold"])
    lines += ["", f"At the cost threshold: {tp} positives caught, {fn} missed, {fp} false alarms, {tn} correct "
              "negatives.", "", "## Permutation importance on the holdout (top 10)", ""]
    lines += [f"- {k}: {v:+.4f}" for k, v in o["permutation importance (log loss)"].items()]
    lines += ["", f"![report](figures/l8_{o['name']}_report.png)", ""]
    (config.REPORTS / f"{o['name']}_report.md").write_text("\n".join(lines))


def german():
    from sklearn.compose import make_column_transformer
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    d = fetch_openml("credit-g", version=1, as_frame=True, data_home=str(config.ROOT / "data" / "sk"), parser="auto")
    X, y = d.data, (d.target == "bad").astype(int).to_numpy()
    cat, num = X.select_dtypes(exclude="number").columns, X.select_dtypes("number").columns
    dense = lambda: make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),  # noqa: E731
                                            ("passthrough", num))
    models = {"logistic": make_pipeline(make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat),
                                                                (StandardScaler(), num)),
                                        LogisticRegression(max_iter=2000)),
              "random forest": make_pipeline(dense(), RandomForestClassifier(500, min_samples_leaf=3,
                                                                             random_state=truth.SEED, n_jobs=-1)),
              "gradient boosting": make_pipeline(dense(), HistGradientBoostingClassifier(
                  max_iter=200, learning_rate=0.05, random_state=truth.SEED))}
    return model_report("german_credit", models, X, y, cost=(5.0, 1.0))


def run():
    o = german()
    print((config.REPORTS / "german_credit_report.md").read_text())
    return o
