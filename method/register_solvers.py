"""Write solver manifests that eval/ discovers in outputs/solvers/.

A manifest is {"command": [...]} where "{sessions}" and "{output}" are
replaced by the evaluator with a session directory and a scores path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]

SOLVERS = {
    "lr": ["--estimator", "lr"],
    "s_lr": ["--estimator", "s_lr"],
    "dummy": ["--estimator", "dummy"],
}


def write_manifest(directory: Path, name: str, extra: list[str]) -> None:
    command = ["python", "method/score_sessions.py", "--sessions", "{sessions}",
               "--output", "{output}", *extra]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps({"command": command}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", choices=sorted(SOLVERS))
    parser.add_argument("--solvers-dir", default=str(WORKSPACE / "outputs/solvers"))
    args = parser.parse_args()
    for name in args.names:
        write_manifest(Path(args.solvers_dir), name, SOLVERS[name])
        print(f"Registered solver {name} in {args.solvers_dir}")


if __name__ == "__main__":
    main()
