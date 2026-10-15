"""Paths, resolved from the project root so commands work from any directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "results"
FIGURES = REPORTS / "figures"
for _d in (RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
