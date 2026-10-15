import numpy as np

from evallab import leakage, truth


def test_logistic_bayes_error_is_below_half():
    e = truth.Logistic().bayes_error(n=20_000)
    assert 0.05 < e < 0.5


def test_noise_wide_is_balanced_and_unrelated():
    X, y = truth.noise_wide(50, 30, np.random.default_rng(0))
    assert X.shape == (50, 30) and y.sum() == 25


def test_overlapping_features_use_only_the_past():
    rng = np.random.default_rng(0)
    X, y, t = truth.overlapping(500, 5, rng)
    # features and targets are built from disjoint days, so across many rows they are uncorrelated
    assert abs(np.corrcoef(X[::10, 0], y[::10])[0, 1]) < 0.25


def test_grouped_rows_share_their_group_signature():
    X, y, g = truth.Grouped(per_group=20).sample(30, np.random.default_rng(0))
    within = np.mean([X[g == i, 1].std() for i in range(30)])
    assert within < X[:, 1].std()


def test_purged_folds_never_train_on_overlapping_targets():
    t = np.arange(100, 400)
    h = 10
    for a, b in leakage.purged_folds(t, h):
        lo, hi = t[b[0]], t[b[-1]]
        ta = t[a]
        assert not np.any((ta + h > lo) & (ta <= hi + h))      # target windows [t, t+h) don't touch the block


def test_walk_forward_trains_only_on_the_past():
    t = np.arange(300)
    for a, b in leakage.walk_forward(t, 5):
        assert t[a].max() + 5 <= t[b].min()


def test_wrong_way_cv_is_optimistic():
    r = leakage.wrong_vs_right(0, n=40, p=2000, k=50)
    assert r["wrong way (select, then CV)"] < 0.3 < r["right way (select inside each fold)"]


def test_skewed_oracles_win_their_own_metric():
    from evallab import regression
    gen = truth.Skewed()
    X, y = gen.sample(200_000, np.random.default_rng(1))
    p = {t: gen.best(X, t) for t in ["mean", "median", "mape", "quantile"]}
    assert regression.METRICS["MAPE"](y, p["mape"]) < regression.METRICS["MAPE"](y, p["median"])
    assert regression.METRICS["MAE"](y, p["median"]) < regression.METRICS["MAE"](y, p["mean"])
    assert regression.METRICS["RMSE"](y, p["mean"]) < regression.METRICS["RMSE"](y, p["median"])
    assert abs(np.mean(p["mean"] - y)) < 0.01 * y.mean()


def test_r2_depends_on_spread_not_error():
    from evallab import regression
    r = regression.r2_range(n=20_000)
    a, b = r.values()
    assert abs(a["RMSE"] - b["RMSE"]) < 0.05 and b["R2"] - a["R2"] > 0.25


def test_with_prevalence_hits_target():
    gen = truth.with_prevalence(0.05)
    _, y = gen.sample(100_000, np.random.default_rng(3))
    assert abs(y.mean() - 0.05) < 0.005


def test_ece_is_zero_for_calibrated_and_large_for_inflated():
    from evallab.classification import ece
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert ece(y, p) < 0.01 and ece(y, np.minimum(p * 3, 1)) > 0.2


def test_views_models_are_symmetric():
    X, y = truth.Views().sample(200_000, np.random.default_rng(0))
    a, b = X[:, :5].sum(1), X[:, 5:].sum(1)
    assert abs(np.corrcoef(a, y)[0, 1] - np.corrcoef(b, y)[0, 1]) < 0.01


def test_slopes_groups_differ():
    X, y, g = truth.Slopes().sample(5, np.random.default_rng(0))
    assert len(np.unique(g)) == 5 and len(y) == 5 * truth.Slopes().per_group


def test_ring_is_not_linear():
    from sklearn.linear_model import LogisticRegression
    gen = truth.Ring()
    X, y = gen.sample(5000, np.random.default_rng(0))
    err = (LogisticRegression().fit(X, y).predict(X) != y).mean()
    assert err > gen.bayes_error(n=50_000) + 0.1


def test_nested_cv_is_less_optimistic_on_noise():
    from evallab import tuning
    r = [tuning.one_5a((s, 10, 0.0, 60)) for s in range(3)]
    claimed = np.mean([x["claimed (best CV score)"] for x in r])
    nested = np.mean([x["nested CV"] for x in r])
    assert claimed < nested                      # the winner's CV error looks better than chance; nested doesn't
