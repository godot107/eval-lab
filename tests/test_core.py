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
