"""Lesson 5: tuning and selection bias.

5a  Pick the best of m configurations by cross-validation, then report that best CV score: the score is the
    *winner* of m noisy estimates, so it is optimistic, and more so as m grows. Nested CV (an outer loop around
    the whole tuning procedure; Witten et al. ch. 5) gives an honest estimate. Truth: the chosen configuration,
    refit on all the data, scored on 100,000 fresh rows. Run on a problem with signal and on pure noise
    (true error 50% whatever is chosen).

5b  Choosing on a single validation group. Groups (machines, sites) bend a shared pattern each in their own way
    (`truth.Slopes`). Choosing the configuration on one held-out group, versus leave-one-group-out CV over all
    groups. Reported: the score each method claims, the chosen configuration's true score on new groups, and the
    regret (how much worse it is than the best configuration in the pool). This is the ast-specialists MIMII
    problem: one machine in validation, another in test.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneGroupOut

from . import config, truth


def configs(m, p, rng):
    """m random configurations: a regularisation strength and a feature subset."""
    return [(float(10 ** rng.uniform(-3, 2)), np.sort(rng.choice(p, rng.integers(2, p + 1), replace=False)))
            for _ in range(m)]


def fit(cfg, X, y):
    C, cols = cfg
    return LogisticRegression(C=C, max_iter=500).fit(X[:, cols], y)


def error(model, cfg, X, y):
    return float((model.predict(X[:, cfg[1]]) != y).mean())


def cv_error(cfg, X, y, folds):
    return float(np.mean([error(fit(cfg, X[a], y[a]), cfg, X[b], y[b]) for a, b in folds]))


def select(cands, X, y, k=5, seed=0):
    folds = list(KFold(k, shuffle=True, random_state=seed).split(X))
    folds = [(a, b) for a, b in folds if len(np.unique(y[a])) == 2]
    errs = [cv_error(c, X, y, folds) for c in cands]
    i = int(np.argmin(errs))
    return cands[i], errs[i]


def one_5a(args):
    seed, m, strength, n = args
    rng = np.random.default_rng(seed)
    gen = truth.Logistic(p=20, k=5, strength=strength)
    X, y = gen.sample(n, rng)
    Xp, yp = gen.sample(20_000, np.random.default_rng(10_000 + seed))
    cands = configs(m, gen.p, rng)
    best, claimed = select(cands, X, y, seed=seed)
    true_err = error(fit(best, X, y), best, Xp, yp)
    outer = []
    for a, b in KFold(5, shuffle=True, random_state=seed + 1).split(X):
        cb, _ = select(cands, X[a], y[a], seed=seed)
        outer.append(error(fit(cb, X[a], y[a]), cb, X[b], y[b]))
    return {"m": m, "problem": "signal" if strength else "pure noise", "claimed (best CV score)": claimed,
            "nested CV": float(np.mean(outer)), "truth": true_err}


def one_5b(args):
    seed, m = args
    gen = truth.Slopes()
    rng = np.random.default_rng(seed)
    X, y, g = gen.sample(12, rng)
    Xp, yp, _ = gen.sample(300, np.random.default_rng(10_000 + seed))
    cands = configs(m, gen.p, rng)
    true = np.array([error(fit(c, X, y), c, Xp, yp) for c in cands])
    val = g == 11                                            # one group held out for validation
    one = [error(fit(c, X[~val], y[~val]), c, X[val], y[val]) for c in cands]
    logo = list(LeaveOneGroupOut().split(X, y, g))
    many = [cv_error(c, X, y, logo) for c in cands]
    out = {}
    for name, s in [("one validation group", one), ("leave-one-group-out CV", many)]:
        i = int(np.argmin(s))
        out[name] = {"claimed": float(s[i]), "truth": float(true[i]), "regret": float(true[i] - true.min())}
    return out


def _pool(fn, jobs, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs, chunksize=2))


def run(sims=40, ms=(1, 10, 100), n=100):
    jobs = [(s, m, st, n) for st in (0.4, 0.0) for m in ms for s in range(sims)]
    a = pd.DataFrame(_pool(one_5a, jobs))
    t5a = a.groupby(["problem", "m"]).mean().reset_index()
    t5a["optimism of claimed"] = t5a["claimed (best CV score)"] - t5a["truth"]
    t5a["optimism of nested"] = t5a["nested CV"] - t5a["truth"]
    b = _pool(one_5b, [(s, 30) for s in range(sims * 2)])
    t5b = {k: pd.DataFrame([r[k] for r in b]).agg(["mean", "std"]).to_dict() for k in b[0]}
    out = {"5a_nested": t5a.to_dict(orient="records"), "5b_one_group": t5b, "sims": sims, "n": n}
    (config.RESULTS / "l5_tuning.json").write_text(json.dumps(out, indent=1))
    a.to_csv(config.RESULTS / "l5a_tuning.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(t5a, "\n", json.dumps(t5b, indent=1))
    plot(t5a, t5b)
    return out


def plot(t5a, t5b):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, prob in zip(axes[:2], ["signal", "pure noise"]):
        d = t5a[t5a.problem == prob]
        x = np.arange(len(d))
        for col, style in [("claimed (best CV score)", "o-"), ("nested CV", "s-"), ("truth", "D--")]:
            ax.plot(x, d[col], style, label=col, color="0.3" if col == "truth" else None)
        ax.set_xticks(x, [f"{m}" for m in d.m])
        ax.set_xlabel("configurations tried")
        ax.set_ylabel("error")
        ax.set_title(f"{prob}: the best CV score is the winner of a lottery", fontsize=10)
        ax.legend(fontsize=8)
    names = list(t5b)
    x = np.arange(len(names))
    claimed = [t5b[n]["claimed"]["mean"] for n in names]
    true = [t5b[n]["truth"]["mean"] for n in names]
    axes[2].bar(x - 0.18, claimed, 0.36, label="claimed")
    axes[2].bar(x + 0.18, true, 0.36, label="truth on new groups")
    axes[2].set_xticks(x, names, fontsize=8)
    axes[2].set_ylabel("error")
    axes[2].set_ylim(0, max(true) * 1.3)
    axes[2].set_title("Choosing on one group vs many", fontsize=10)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "l5_tuning.png", dpi=130)
    plt.close(fig)
