"""Write solver registries for the hardening audit (reference, variants, shortcuts)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SCORE = ["python", "method/score_sessions.py", "--sessions", "{sessions}", "--output", "{output}"]
SHORT = ["python", "redteam/hardening/shortcuts.py", "--sessions", "{sessions}", "--output", "{output}"]

REGISTRIES = {
    "reference": {
        "lr": SCORE + ["--estimator", "lr"],
        "s_lr": SCORE + ["--estimator", "s_lr"],
        "dummy": SCORE + ["--estimator", "dummy"],
    },
    "variants": {
        "lr_seed1": SCORE + ["--estimator", "lr", "--seed", "1"],
        "lr_seed2": SCORE + ["--estimator", "lr", "--seed", "2"],
        "better_cv100": SCORE + ["--estimator", "lr", "--n-splits", "100"],
        "better_smooth": SCORE + ["--estimator", "lr", "--smooth"],
        "better_cv100_smooth": SCORE + ["--estimator", "lr", "--n-splits", "100", "--smooth"],
        "weaker_quarter_data": SCORE + ["--estimator", "lr", "--subsample", "0.25"],
    },
    "light_variants": {
        "better_smooth": SCORE + ["--estimator", "lr", "--smooth"],
        "weaker_quarter_data": SCORE + ["--estimator", "lr", "--subsample", "0.25"],
    },
    "shortcuts": {
        f"shortcut_{kind}": SHORT + ["--kind", kind]
        for kind in ["centroid", "density", "catalogue", "eeg_lookup", "constant", "nan",
                     "missing", "scorefile_forgery", "hidden_oracle"]
    },
    "light_shortcuts": {
        f"shortcut_{kind}": SHORT + ["--kind", kind] for kind in ["centroid", "density"]
    },
}


def main() -> None:
    out = Path(sys.argv[1])
    for group, solvers in REGISTRIES.items():
        directory = out / group
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.json"):
            old.unlink()
        for name, command in solvers.items():
            (directory / f"{name}.json").write_text(json.dumps({"command": command}, indent=1))
    print(f"wrote registries under {out}")


if __name__ == "__main__":
    main()
