"""Register the paper's control conditions as solvers.

s_lr: CURSOR with stimulus/EEG pairs shuffled before scoring (paper's S-LR).
dummy: CURSOR with a mean predictor that ignores the EEG (paper's Dummy).
Both run through method/score_sessions.py and are scored by eval/ exactly like
any other solver.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[1]


def main() -> None:
    names = sys.argv[1:] or ["s_lr", "dummy"]
    subprocess.run([sys.executable, str(WORKSPACE / "method/register_solvers.py"), *names], check=True)


if __name__ == "__main__":
    main()
