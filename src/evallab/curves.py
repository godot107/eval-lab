"""Lesson 7: learning curves and validation curves, read against the truth.

A curved class boundary (`truth.Ring`) and three models:
  logistic regression      can't draw a circle: high bias. Training and CV error meet early, well above the floor;
                           more data doesn't help, a better model does.
  1-nearest neighbour      memorises: training error 0, a big gap to CV error (high variance). More data helps.
  15-nearest neighbours    in between.

For each training size: training error, 5-fold CV error and the true error on fresh data (all averaged over draws),
with the best possible (Bayes) error as the floor. Then a validation curve: a decision tree's depth from 1 to 15
at 400 rows, with the depth CV would pick and the depth the truth would pick.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config, truth

SIZES = [50, 100, 200, 400, 800, 1600, 3200]
MODELS = {"logistic regression (high bias)": LogisticRegression(max_iter=1000),
          "1-nearest neighbour (high variance)": KNeighborsClassifier(1),
          "15-nearest neighbours": KNeighborsClassifier(15)}


def point(args):
    seed, n, name, model = args
    gen = truth.Ring()
    X, y = gen.sample(n, np.random.default_rng(seed))
    Xp, yp = gen.sample(20_000, np.random.default_rng(10_000 + seed))
    m = clone(model).fit(X, y)
    return {"model": name, "n": n, "training": float((m.predict(X) != y).mean()),
            "CV": float(1 - cross_val_score(clone(model), X, y, cv=5).mean()),
            "truth": float((m.predict(Xp) != yp).mean())}


def depth_point(args):
    seed, depth, n = args
    gen = truth.Ring()
    X, y = gen.sample(n, np.random.default_rng(seed))
    Xp, yp = gen.sample(20_000, np.random.default_rng(10_000 + seed))
    model = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    m = clone(model).fit(X, y)
    return {"depth": depth, "training": float((m.predict(X) != y).mean()),
            "CV": float(1 - cross_val_score(clone(model), X, y, cv=5).mean()),
            "truth": float((m.predict(Xp) != yp).mean())}


def run(draws=10, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(s, n, k, m) for k, m in MODELS.items() for n in SIZES for s in range(draws)]
    djobs = [(s, d, 400) for d in range(1, 16) for s in range(draws * 2)]
    with ProcessPoolExecutor(workers) as ex:
        lc = pd.DataFrame(list(ex.map(point, jobs, chunksize=4)))
        vc = pd.DataFrame(list(ex.map(depth_point, djobs, chunksize=4)))
    lc = lc.groupby(["model", "n"]).mean().reset_index()
    vc = vc.groupby("depth").mean().reset_index()
    bayes = truth.Ring().bayes_error()
    out = {"bayes_error": bayes, "learning": lc.to_dict(orient="records"), "validation": vc.to_dict(orient="records"),
           "depth chosen by CV": int(vc.loc[vc.CV.idxmin(), "depth"]),
           "depth best on truth": int(vc.loc[vc.truth.idxmin(), "depth"])}
    (config.RESULTS / "l7_curves.json").write_text(json.dumps(out, indent=1))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(lc.pivot(index="n", columns="model", values=["training", "CV", "truth"]), "\n", vc,
              "\nbayes", round(bayes, 3))
    plot(lc, vc, bayes)
    return out


def plot(lc, vc, bayes):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, name in zip(axes, MODELS):
        d = lc[lc.model == name]
        for col, style in [("training", "o-"), ("CV", "s-"), ("truth", "D--")]:
            ax.plot(d.n, d[col], style, ms=4, label=col, color="0.3" if col == "truth" else None)
        ax.axhline(bayes, color="#b8422e", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("training rows")
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel("error")
    axes[0].text(55, bayes + 0.01, "best possible", color="#b8422e", fontsize=8)
    axes[0].legend(fontsize=8)
    for col, style in [("training", "o-"), ("CV", "s-"), ("truth", "D--")]:
        axes[3].plot(vc.depth, vc[col], style, ms=4, label=col, color="0.3" if col == "truth" else None)
    axes[3].axhline(bayes, color="#b8422e", lw=1)
    axes[3].set_xlabel("tree depth (400 rows)")
    axes[3].set_title("Validation curve: complexity", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "l7_curves.png", dpi=130)
    plt.close(fig)
