"""python -m evallab <lesson> [<lesson> ...]

  classification  lesson 1: imbalance, calibration, cost-based thresholds, German credit
  regression      lesson 2: which metric, R^2's dependence on spread, interval coverage, California housing
  estimators      lesson 3: holdout vs K-fold vs repeated CV vs .632 bootstrap, against the true error
  leakage         lesson 4: selection before CV, grouped records, overlapping time targets
  purge           lesson 4c follow-up: when purging matters (folds x horizon)
  tuning          lesson 5: nested vs non-nested CV, choosing on one validation group
  compare         lesson 6: false-positive rate and power of tests for "B beats A"
  curves          lesson 7: learning and validation curves
  report          lesson 8: model-quality report on German credit
  all             every lesson, in order
"""
import importlib
import os
import sys

# Simulations run in a process pool; one math-library thread per process, or 8 workers x 12 threads fight over
# 12 cores (load average 54 on a first run of lesson 5).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

LESSONS = {"classification": ("classification", "run"), "regression": ("regression", "run"),
           "estimators": ("estimators", "run"), "leakage": ("leakage", "run"), "purge": ("leakage", "purge_sweep"),
           "tuning": ("tuning", "run"), "compare": ("compare", "run"), "curves": ("curves", "run"),
           "report": ("report", "run")}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    names = list(LESSONS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for name in names:
        if name not in LESSONS:
            sys.exit(f"unknown lesson {name!r}\n{__doc__}")
        mod, fn = LESSONS[name]
        print(f"== {name}", flush=True)
        getattr(importlib.import_module(f"evallab.{mod}"), fn)()


if __name__ == "__main__":
    main()
