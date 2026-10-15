"""python -m evallab <lesson>

  regression   lesson 2: which metric, R^2's dependence on spread, interval coverage, California housing
  estimators   lesson 3: holdout vs K-fold vs repeated CV vs .632 bootstrap, against the true error
  leakage      lesson 4: selection before CV, grouped records, overlapping time targets
"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "regression":
        from . import regression
        regression.run()
    elif cmd == "estimators":
        from . import estimators
        print(estimators.run())
    elif cmd == "leakage":
        from . import leakage
        leakage.run()
    else:
        sys.exit(f"unknown lesson {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
