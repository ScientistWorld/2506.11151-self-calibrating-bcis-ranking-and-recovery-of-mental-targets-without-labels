from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


WORKSPACE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CURSOR control baselines.")
    parser.add_argument("--output", default=str(WORKSPACE / "outputs/baseline_ranking.json"))
    parser.add_argument("--n-targets", type=int, default=3)
    parser.add_argument("--n-observations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cmd = [
        "python",
        str(WORKSPACE / "method/run_cursor.py"),
        "--methods",
        "s_lr",
        "dummy",
        "--output",
        args.output,
        "--n-targets",
        str(args.n_targets),
        "--n-observations",
        str(args.n_observations),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
