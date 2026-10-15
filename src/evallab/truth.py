"""Data generators whose truth is known.

Real datasets can't tell you how wrong an evaluation was, because the true score on future data is unknown.
Here every dataset comes from a generator, so a fitted model can also be scored on a large fresh sample from the
same generator: that score is the truth each evaluation method is trying to estimate.

Each generator returns (X, y) for a training set and has a matching way to draw fresh data:
- `Logistic`      a plain binary classification problem (lesson 3: how good is each error estimate?)
- `noise_wide`    many features, none related to the label (lesson 4a: the wrong way to cross-validate)
- `Grouped`       repeated records per patient/machine with a patient signature (lesson 4b: grouped leakage)
- `overlapping`   a time series whose targets overlap in time and carry no signal (lesson 4c: time leakage)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEED = 42
POP = 100_000        # fresh draws used as "the truth"


@dataclass
class Logistic:
    """P(y=1|x) = sigmoid(b0 + x[:, :k] . beta); the other p - k features are noise."""
    p: int = 20
    k: int = 5
    strength: float = 0.8
    b0: float = 0.0

    def __post_init__(self):
        self.beta = np.full(self.k, self.strength)

    def prob(self, X):
        return 1 / (1 + np.exp(-(self.b0 + X[:, :self.k] @ self.beta)))

    def sample(self, n, rng):
        X = rng.standard_normal((n, self.p))
        return X, (rng.random(n) < self.prob(X)).astype(int)

    def bayes_error(self, rng=None, n=POP):
        """Error of the best possible classifier (predict the more likely class): the floor no model beats."""
        rng = rng or np.random.default_rng(SEED)
        X = rng.standard_normal((n, self.p))
        q = self.prob(X)
        return float(np.minimum(q, 1 - q).mean())


def noise_wide(n, p, rng):
    """ESL 7.10.2: balanced labels and p standard-normal features, all independent of the label.
    Every classifier's true error is exactly 50%."""
    y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    return rng.standard_normal((n, p)), rng.permutation(y)


@dataclass
class Grouped:
    """Several records per group (patient, machine, customer).

    Each group has a hidden risk u_g that shifts its label, and a feature 'signature' s_g added to all of its
    records (same device, same body, same site). A flexible model can learn "this signature means this risk",
    which is true for groups it has seen and useless for new ones. Only x[:, 0] carries signal that transfers.
    """
    p: int = 10
    per_group: int = 10
    risk_sd: float = 2.0
    signature_sd: float = 1.0
    slope: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False)

    def sample(self, n_groups, rng):
        g = np.repeat(np.arange(n_groups), self.per_group)
        u = rng.normal(0, self.risk_sd, n_groups)[g]
        X = rng.normal(0, self.signature_sd, (n_groups, self.p))[g] + rng.standard_normal((len(g), self.p))
        logit = self.slope * X[:, 0] + u
        y = (rng.random(len(g)) < 1 / (1 + np.exp(-logit))).astype(int)
        return X, y, g


def overlapping(n, h, rng, windows=(20, 60)):
    """Daily 'returns' r_t are pure noise, so nothing predicts the future: the true R^2 of any model is <= 0.

    Target y_t = r_{t+1} + ... + r_{t+h} (the next h days), so neighbouring targets share h - 1 terms.
    Features are trailing means of r over long windows, so neighbouring rows also look almost identical.
    Shuffle such rows into folds and a model can 'predict' a test row from its neighbour in training.
    Returns X (n, len(windows)), y (n,), and t (the row's time index).
    """
    w = max(windows)
    r = rng.standard_normal(n + w + h)
    c = np.r_[0, np.cumsum(r)]
    t = np.arange(w, w + n)
    X = np.c_[[(c[t] - c[t - k]) / k for k in windows]].T
    y = c[t + h] - c[t]
    return X, y, t
