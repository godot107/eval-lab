"""Lesson 1: choosing the classification metric.

1a  Imbalance. The same kind of problem at 50%, 10% and 1% positives. Accuracy rewards a model that never predicts
    the rare class; ROC AUC stays flat as positives get rarer while PR AUC (average precision) falls toward the
    prevalence, because precision is what suffers when positives are rare. Brier and log loss score the
    probabilities themselves; a Brier *skill* score compares with always predicting the prevalence.

1b  Calibration. Four models on a 10%-positive problem: plain logistic regression, the same with
    class_weight="balanced" (a common fix for imbalance), gradient boosting, and the balanced model recalibrated
    (isotonic) on a held-out split. Ranking metrics (ROC AUC) can't see that balanced weights inflate every
    probability; proper scores (log loss, Brier) and the calibration error can.

1c  Thresholds from costs. If missing a positive costs 5 and a false alarm costs 1 (the German credit cost
    matrix), the best threshold on *calibrated* probabilities is 1 / (1 + 5) = 0.167, not 0.5. Expected cost per
    case on fresh data for: 0.5, the F1-maximising threshold, 0.167, and 0.167 applied to the balanced model.

1d  Real data: German credit (1,000 loans, 30% bad, cost 5 for approving a bad loan, 1 for rejecting a good one),
    5-fold CV, the same threshold comparison.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, truth

C_FN, C_FP = 5.0, 1.0                       # German credit: approving a bad loan costs 5x rejecting a good one
COST_THRESHOLD = C_FP / (C_FP + C_FN)       # predict positive when P(positive) > 1/6


def ece(y, p, bins=10):
    """Expected calibration error: average |observed rate - mean prediction| over equal-count bins."""
    order = np.argsort(p)
    return float(sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in np.array_split(order, bins)))


def probs_scores(y, p, thr=0.5):
    base = np.full(len(y), y.mean())
    return {"accuracy": accuracy_score(y, p > thr), "balanced accuracy": balanced_accuracy_score(y, p > thr),
            "ROC AUC": roc_auc_score(y, p), "PR AUC": average_precision_score(y, p),
            "log loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "Brier": brier_score_loss(y, p),
            "Brier skill": 1 - brier_score_loss(y, p) / brier_score_loss(y, base),
            "calibration error": ece(y, p), "mean prediction": float(p.mean()), "prevalence": float(y.mean())}


def cost(y, pred):
    return float((C_FN * ((pred == 0) & (y == 1)) + C_FP * ((pred == 1) & (y == 0))).mean())


def f1_threshold(y, p):
    pr, rc, t = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(t[np.argmax(f[:-1])])


def imbalance(n=5000):
    rows = {}
    for prev in [0.5, 0.1, 0.01]:
        gen = truth.with_prevalence(prev)
        rng = np.random.default_rng(truth.SEED)
        X, y = gen.sample(n, rng)
        Xp, yp = gen.sample(truth.POP, rng)
        p = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(Xp)[:, 1]
        rows[f"{prev:.0%} positives: model"] = probs_scores(yp, p)
        rows[f"{prev:.0%} positives: always 'no'"] = {**probs_scores(yp, np.full(len(yp), 1e-3)),
                                                     "ROC AUC": 0.5, "PR AUC": float(yp.mean())}
        rows[f"{prev:.0%} positives: true probabilities"] = probs_scores(yp, gen.prob(Xp))
    return pd.DataFrame(rows).T


def calibration(n=5000, prev=0.1):
    gen = truth.with_prevalence(prev)
    rng = np.random.default_rng(truth.SEED)
    X, y = gen.sample(n, rng)
    Xp, yp = gen.sample(truth.POP, rng)
    Xa, Xc, ya, yc = train_test_split(X, y, test_size=0.3, random_state=truth.SEED, stratify=y)
    plain = LogisticRegression(max_iter=1000).fit(Xa, ya)
    bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xa, ya)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=truth.SEED).fit(Xa, ya)
    recal = CalibratedClassifierCV(FrozenEstimator(bal), method="isotonic").fit(Xc, yc)
    models = {"logistic": plain, "logistic, balanced weights": bal, "gradient boosting": gbm,
              "balanced, recalibrated": recal}
    P = {k: m.predict_proba(Xp)[:, 1] for k, m in models.items()}
    P["true probabilities"] = gen.prob(Xp)
    table = pd.DataFrame({k: probs_scores(yp, p) for k, p in P.items()}).T

    # 1c: thresholds; the F1 threshold is chosen on the calibration split, never on the fresh data
    th = {"0.5 (default)": ("logistic", 0.5),
          "F1-maximising (chosen on held-out split)": ("logistic", f1_threshold(yc, plain.predict_proba(Xc)[:, 1])),
          f"cost-based 1/(1+{C_FN:.0f}) = {COST_THRESHOLD:.3f}": ("logistic", COST_THRESHOLD),
          f"cost-based {COST_THRESHOLD:.3f} on the balanced model": ("logistic, balanced weights", COST_THRESHOLD),
          f"cost-based {COST_THRESHOLD:.3f} on true probabilities": ("true probabilities", COST_THRESHOLD)}
    costs = {name: {"threshold": t, "expected cost per case": cost(yp, (P[m] > t).astype(int)),
                    "share flagged": float((P[m] > t).mean())} for name, (m, t) in th.items()}
    curves = {k: [list(map(float, a)) for a in calibration_curve(yp, P[k], n_bins=10, strategy="quantile")]
              for k in P}
    return table, costs, curves


def german(k=5):
    from sklearn.datasets import fetch_openml
    d = fetch_openml("credit-g", version=1, as_frame=True, data_home=str(config.ROOT / "data" / "sk"),
                     parser="auto")
    X, y = d.data, (d.target == "bad").astype(int).to_numpy()
    cat = X.select_dtypes(exclude="number").columns
    num = X.select_dtypes("number").columns
    pre = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat), (StandardScaler(), num))
    models = {"logistic": make_pipeline(pre, LogisticRegression(max_iter=2000)),
              "gradient boosting": make_pipeline(make_column_transformer(
                  (OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat), ("passthrough", num)),
                  HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=truth.SEED))}
    out = {}
    for name, m in models.items():
        p, f1t = np.zeros(len(y)), []
        for a, b in StratifiedKFold(k, shuffle=True, random_state=truth.SEED).split(X, y):
            fit = clone(m).fit(X.iloc[a], y[a])
            p[b] = fit.predict_proba(X.iloc[b])[:, 1]
        s = probs_scores(y, p)
        out[name] = {"ROC AUC": s["ROC AUC"], "Brier": s["Brier"], "calibration error": s["calibration error"],
                     "accuracy at 0.5": s["accuracy"],
                     "cost per loan at 0.5": cost(y, (p > 0.5).astype(int)),
                     f"cost per loan at {COST_THRESHOLD:.3f}": cost(y, (p > COST_THRESHOLD).astype(int)),
                     "cost per loan, approve everyone": cost(y, np.zeros(len(y), int)),
                     "cost per loan, reject everyone": cost(y, np.ones(len(y), int))}
    return pd.DataFrame(out).T, {"n": int(len(y)), "bad_rate": float(y.mean())}


def run():
    a = imbalance()
    b, c, curves = calibration()
    d, info = german()
    out = {"1a_imbalance": a.to_dict(orient="index"), "1b_calibration": b.to_dict(orient="index"),
           "1c_thresholds": c, "1b_curves": curves, "1d_german": d.to_dict(orient="index"), "1d_info": info}
    (config.RESULTS / "l1_classification.json").write_text(json.dumps(out, indent=1))
    with pd.option_context("display.width", 200, "display.precision", 3, "display.max_columns", 20):
        print(a, "\n\n", b, "\n\n", pd.DataFrame(c).T, "\n\n", d)
    plot(a, curves, c)
    return out


def plot(a, curves, c):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    prevs = ["50%", "10%", "1%"]
    for metric, ls in [("accuracy", "-"), ("ROC AUC", "-"), ("PR AUC", "-")]:
        axes[0].plot(prevs, [a.loc[f"{p} positives: model", metric] for p in prevs], marker="o", ls=ls, label=metric)
    axes[0].plot(prevs, [a.loc[f"{p} positives: always 'no'", "accuracy"] for p in prevs], marker="x", ls="--",
                 color="0.4", label="accuracy of always 'no'")
    axes[0].set_xlabel("share of positives")
    axes[0].set_title("Same model quality, rarer positives", fontsize=10)
    axes[0].legend(fontsize=8)
    for k, (frac, mean) in curves.items():
        if k == "true probabilities":
            continue
        axes[1].plot(mean, frac, marker="o", ms=3, label=k)
    axes[1].plot([0, 1], [0, 1], color="0.5", lw=0.8)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("predicted probability")
    axes[1].set_ylabel("observed share positive")
    axes[1].set_title("Calibration (10% positives): on the line = honest probabilities", fontsize=10)
    axes[1].legend(fontsize=8)
    names = list(c)
    axes[2].barh(range(len(names))[::-1], [c[n]["expected cost per case"] for n in names], color="#2a6fbb")
    axes[2].set_yticks(range(len(names))[::-1], [n.replace(" (chosen on held-out split)", "") for n in names],
                       fontsize=8)
    axes[2].set_xlabel(f"cost per case (miss {C_FN:.0f}, false alarm {C_FP:.0f})")
    axes[2].set_title("Which threshold?", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "l1_classification.png", dpi=130)
    plt.close(fig)
