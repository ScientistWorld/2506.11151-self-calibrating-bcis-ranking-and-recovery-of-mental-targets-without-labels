"""Train-slice (visible development) evaluator for label-free mental-target ranking.

This file is an independent copy of the session generator and scorer used by
the hidden test; it shares no code with ``eval/test/``.  It differs only in
its split seed (sessions here are visible and may be exported together with
their answers for development via ``--export-sessions``).

The evaluator owns every input of the comparison: it generates sessions
(stimulus latents, EEG responses, 60 candidates with one target), runs every
registered solver (``outputs/solvers/*.json``) on the same sessions, reads back
only raw candidate scores, and recomputes every metric itself.

Stimuli and candidates are drawn independently of the target from one shared
distribution in a random low-dimensional affine subspace of the 512-D latent
space, and the target is a uniformly random candidate, so the geometry alone
does not identify the target.  EEG responses are real recorded rows whose
recorded stimulus-target distance matches each stimulus's distance to the
target; each session re-standardises the EEG features and rotates them by a
random orthogonal matrix.  Control sessions permute EEG rows across stimuli;
a solver that ranks far better than chance on them fails the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
SLICE_NAME = "train"
SPLIT_SEED = 20250611
N_ALIGNED = 48
N_CONTROL = 16
N_CANDIDATES = 60
LATENT_DIM = 512
NEAR_TARGET_DISTANCE = 5.0
CONTROL_GATE_SE = 4.0
DESIGNS = {
    # latent_dim: dimension of the affine subspace holding stimuli and
    # candidates; spread: per-axis standard deviation (latent units);
    # n_observations: stimulus/EEG pairs per session.
    "k2": {"latent_dim": 2, "spread": 13.0, "n_observations": 3000},
    "k2_dense": {"latent_dim": 2, "spread": 9.0, "n_observations": 3000},
    "k2_n6000": {"latent_dim": 2, "spread": 13.0, "n_observations": 6000},
    "k3": {"latent_dim": 3, "spread": 10.2, "n_observations": 3000},
    "k5": {"latent_dim": 5, "spread": 7.65, "n_observations": 3000},
    # Homogeneous design: stimuli and candidates uniform on a 2-sphere of
    # radius 17.3 (random 3-D subspace).  No candidate is a priori closer to a
    # random target; mean candidate-target distance 23.07 (paper chance 23.08).
    "s2": {"latent_dim": 3, "manifold": "sphere", "radius": 17.3, "n_observations": 3000},
}
DEFAULT_DESIGN = "s2"
EXPERIMENTS = {
    "unbiased_ranking_raw_eeg": {"representation": "eeg_raw", "session_set": "main"},
    "unbiased_ranking_eegnet": {"representation": "eeg_net", "session_set": "main"},
    # The paper's data-size axis: independent sessions with fewer observations.
    "unbiased_ranking_raw_eeg_low_data": {"representation": "eeg_raw", "session_set": "low_data"},
}
REPRESENTATIONS = sorted({spec["representation"] for spec in EXPERIMENTS.values()})
LOW_DATA_OBSERVATIONS = 1000
METRICS = (
    "score_distance_pearson_r",
    "target_rank",
    "top_rank_distance",
    "target_top1_accuracy",
    "target_top5_accuracy",
    "reciprocal_target_rank",
    "near_target_top_rank_rate",
)
DEFAULT_TIMEOUT = float(os.environ.get("CURSOR_SOLVER_TIMEOUT", 3 * 3600))


# --------------------------------------------------------------------------
# Session generation
# --------------------------------------------------------------------------


def load_pool(path: Path) -> dict:
    with np.load(path) as arrays:
        target = np.asarray(arrays["target_faces"], dtype=float)
        observed = np.asarray(arrays["observed_faces"], dtype=float)
        eeg = {
            "eeg_raw": np.asarray(arrays["eeg_raw"], dtype=float),
            "eeg_net": np.asarray(arrays["eeg_net"], dtype=float),
        }
    recorded = np.linalg.norm(observed - target, axis=1)
    levels, level_of_row = np.unique(np.round(recorded, 2), return_inverse=True)
    rows_by_level = [np.flatnonzero(level_of_row == j) for j in range(len(levels))]
    return {"levels": levels, "rows_by_level": rows_by_level, "eeg": eeg}


def _assign_rows(distances: np.ndarray, pool: dict, rng: np.random.Generator) -> np.ndarray:
    """Pick one recorded EEG row per stimulus at the nearest recorded distance.

    Rows are used at most once per session; when a level runs out, the
    stimulus takes an unused row from the nearest level that still has one.
    """
    levels = pool["levels"]
    remaining = [list(rng.permutation(rows)) for rows in pool["rows_by_level"]]
    rows = np.empty(len(distances), dtype=int)
    for i in rng.permutation(len(distances)):
        for level in np.argsort(np.abs(levels - distances[i]), kind="stable"):
            if remaining[level]:
                rows[i] = remaining[level].pop()
                break
        else:  # pragma: no cover - sessions are far smaller than the pool
            raise RuntimeError("EEG pool exhausted")
    return rows


def _session_basis_eeg(eeg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mean = eeg.mean(axis=0)
    scale = eeg.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (eeg - mean) / scale
    q, r = np.linalg.qr(rng.normal(size=(eeg.shape[1], eeg.shape[1])))
    q *= np.sign(np.diag(r))
    return (standardized @ q).astype(np.float32)


def _sample_latents(rng: np.random.Generator, n: int, design: dict) -> np.ndarray:
    if design.get("manifold") == "sphere":
        x = rng.normal(size=(n, design["latent_dim"]))
        return design["radius"] * x / np.linalg.norm(x, axis=1, keepdims=True)
    return rng.normal(scale=design["spread"], size=(n, design["latent_dim"]))


def make_sessions(pool: dict, design_name: str, seed: int, n_aligned: int,
                  n_control: int, n_observations: int | None = None) -> list[dict]:
    design = dict(DESIGNS[design_name])
    if n_observations is not None:
        design["n_observations"] = n_observations
    k = design["latent_dim"]
    n_obs = design["n_observations"]
    design_code = int(hashlib.sha256(design_name.encode()).hexdigest()[:8], 16)
    root = np.random.SeedSequence([seed, design_code])
    sessions = []
    for index, child in enumerate(root.spawn(n_aligned + n_control)):
        rng = np.random.default_rng(child)
        center = rng.normal(size=LATENT_DIM)
        basis, _ = np.linalg.qr(rng.normal(size=(LATENT_DIM, k)))
        obs = _sample_latents(rng, n_obs, design)
        cand = _sample_latents(rng, N_CANDIDATES, design)
        target = int(rng.integers(N_CANDIDATES))
        rows = _assign_rows(np.linalg.norm(obs - cand[target], axis=1), pool, rng)
        control = index >= n_aligned
        if control:
            rows = rows[rng.permutation(n_obs)]
        eeg_seeds = rng.integers(0, 2**63 - 1, size=len(REPRESENTATIONS))
        token = hashlib.sha256(f"{SLICE_NAME}:{seed}:{design_name}:{index}".encode()).hexdigest()
        sessions.append(
            {
                "id": "s" + token[:12],
                "control": control,
                "center": center,
                "basis": basis,
                "obs": obs,
                "cand": cand,
                "target": target,
                "true_distances": np.linalg.norm(cand - cand[target], axis=1),
                "rows": rows,
                "eeg_seeds": dict(zip(REPRESENTATIONS, eeg_seeds)),
            }
        )
    return sessions


def write_sessions(sessions: list[dict], pool: dict, representation: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        rng = np.random.default_rng(int(session["eeg_seeds"][representation]))
        eeg = _session_basis_eeg(pool["eeg"][representation][session["rows"]], rng)
        stimuli = session["center"] + session["obs"] @ session["basis"].T
        candidates = session["center"] + session["cand"] @ session["basis"].T
        np.savez(
            out_dir / f"{session['id']}.npz",
            stimuli=stimuli.astype(np.float32),
            eeg=eeg,
            candidates=candidates.astype(np.float32),
        )
    index = {
        "format": "cursor-sessions-v1",
        "eeg_representation": representation,
        "n_candidates": N_CANDIDATES,
        "sessions": sorted(session["id"] for session in sessions),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))


def export_sessions(session_sets: dict, pool: dict, out_dir: Path) -> None:
    for experiment, spec in EXPERIMENTS.items():
        write_sessions(session_sets[spec["session_set"]], pool, spec["representation"], out_dir / experiment)
    sessions = [s for group in session_sets.values() for s in group]
    truth = {
        session["id"]: {
            "control": bool(session["control"]),
            "target_index": int(session["target"]),
            "true_distances": session["true_distances"].tolist(),
        }
        for session in sessions
    }
    (out_dir / "train_answers.json").write_text(json.dumps(truth))
    print(f"Exported {len(sessions)} train sessions to {out_dir}")


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------


def load_solvers(solvers_dir: Path) -> list[tuple[str, list[str]]]:
    solvers = []
    for manifest in sorted(Path(solvers_dir).glob("*.json")):
        name = manifest.stem
        if name.startswith("_"):
            continue
        spec = json.loads(manifest.read_text())
        command = spec.get("command")
        if not isinstance(command, list) or not all(isinstance(a, str) for a in command):
            raise ValueError(f"{manifest}: 'command' must be a list of strings")
        solvers.append((name, command))
    return solvers


def run_solver(name: str, command: list[str], sessions_dir: Path, output: Path,
               log: Path, timeout: float) -> str | None:
    args = [a.replace("{sessions}", str(sessions_dir)).replace("{output}", str(output)) for a in command]
    if args and args[0] in {"python", "python3"}:
        args[0] = sys.executable
    started = time.time()
    try:
        with open(log, "w") as handle:
            proc = subprocess.run(args, cwd=WORKSPACE, stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout:.0f}s"
    except OSError as error:
        return f"could not start: {error}"
    print(f"[{SLICE_NAME}] solver {name} finished in {time.time() - started:.1f}s "
          f"(exit {proc.returncode})", flush=True)
    if proc.returncode != 0:
        tail = log.read_text(errors="replace")[-2000:]
        print(tail, flush=True)
        return f"exit status {proc.returncode}"
    return None


def read_scores(output: Path, sessions: list[dict]) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(output.read_text())
        raw = payload["scores"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"unreadable scores file: {error!r}"
    scores = {}
    for session in sessions:
        values = raw.get(session["id"]) if isinstance(raw, dict) else None
        if values is None:
            return None, f"missing scores for session {session['id']}"
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return None, f"non-numeric scores for session {session['id']}"
        if array.shape != (N_CANDIDATES,):
            return None, f"session {session['id']} has shape {array.shape}, expected ({N_CANDIDATES},)"
        if not np.all(np.isfinite(array)):
            return None, f"non-finite scores for session {session['id']}"
        scores[session["id"]] = array
    return scores, None


# --------------------------------------------------------------------------
# Metrics (expected values under uniformly random tie-breaking)
# --------------------------------------------------------------------------


def session_metrics(scores: np.ndarray, distances: np.ndarray, target: int) -> dict[str, float]:
    target_score = scores[target]
    better = int(np.sum(scores > target_score))
    tied = int(np.sum(scores == target_score)) - 1
    positions = np.arange(better + 1, better + tied + 2, dtype=float)
    top = np.flatnonzero(scores == scores.max())
    if np.std(scores) > 0:
        r = float(np.corrcoef(scores, distances)[0, 1])
    else:
        r = 0.0
    return {
        "score_distance_pearson_r": r,
        "target_rank": float(positions.mean()),
        "top_rank_distance": float(distances[top].mean()),
        "target_top1_accuracy": float(np.mean(positions <= 1)),
        "target_top5_accuracy": float(np.mean(positions <= 5)),
        "reciprocal_target_rank": float(np.mean(1.0 / positions)),
        "near_target_top_rank_rate": float(np.mean(distances[top] <= NEAR_TARGET_DISTANCE)),
    }


def failed_metrics(sessions: list[dict]) -> dict[str, float]:
    aligned = [s for s in sessions if not s["control"]]
    return {
        "score_distance_pearson_r": 1.0,
        "target_rank": float(N_CANDIDATES),
        "top_rank_distance": float(np.mean([s["true_distances"].max() for s in aligned])),
        "target_top1_accuracy": 0.0,
        "target_top5_accuracy": 0.0,
        "reciprocal_target_rank": 1.0 / N_CANDIDATES,
        "near_target_top_rank_rate": 0.0,
    }


def summarize(scores: dict, sessions: list[dict]) -> tuple[dict, dict, float]:
    per_session = {metric: [] for metric in METRICS}
    control_ranks = []
    for session in sorted(sessions, key=lambda s: s["id"]):
        values = session_metrics(scores[session["id"]], session["true_distances"], session["target"])
        if session["control"]:
            control_ranks.append(values["target_rank"])
            continue
        for metric in METRICS:
            per_session[metric].append(values[metric])
    means = {metric: float(np.mean(per_session[metric])) for metric in METRICS}
    control_mean = float(np.mean(control_ranks)) if control_ranks else float("nan")
    return means, per_session, control_mean


def control_gate_threshold(n_control: int) -> float:
    chance = (N_CANDIDATES + 1) / 2
    sd = math.sqrt((N_CANDIDATES**2 - 1) / 12)
    return chance - CONTROL_GATE_SE * sd / math.sqrt(max(n_control, 1))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Score registered solvers on visible train sessions.")
    parser.add_argument("--solvers-dir", default=str(WORKSPACE / "outputs/solvers"))
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring/scores_train.json"))
    parser.add_argument("--data", default=str(WORKSPACE / "data/downloads/all_data_sorted.npz"))
    parser.add_argument("--design", default=DEFAULT_DESIGN, choices=sorted(DESIGNS))
    parser.add_argument("--per-session-out", default=None,
                        help="Optional JSON file for per-session metrics (analysis only).")
    parser.add_argument("--workdir", default=str(WORKSPACE / "outputs/.eval_scratch"),
                        help="Scratch directory for session files (removed afterwards).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--export-sessions", default=None,
                        help="Write the visible train sessions (and their answers) to this "
                             "directory and exit without running solvers.")
    # Smoke-test overrides; the official test uses the defaults.
    parser.add_argument("--n-aligned", type=int, default=N_ALIGNED)
    parser.add_argument("--n-control", type=int, default=N_CONTROL)
    parser.add_argument("--n-observations", type=int, default=None)
    parser.add_argument("--low-data-observations", type=int, default=LOW_DATA_OBSERVATIONS)
    args = parser.parse_args()

    pool = load_pool(Path(args.data))
    session_sets = {
        "main": make_sessions(pool, args.design, SPLIT_SEED, args.n_aligned, args.n_control,
                              args.n_observations),
        "low_data": make_sessions(pool, args.design, SPLIT_SEED + 1, args.n_aligned, args.n_control,
                                  args.low_data_observations),
    }
    if args.export_sessions:
        export_sessions(session_sets, pool, Path(args.export_sessions))
        return
    solvers = load_solvers(Path(args.solvers_dir))
    if not solvers:
        raise SystemExit(f"No solver manifests found in {args.solvers_dir}")
    threshold = control_gate_threshold(args.n_control)

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"cursor_{SLICE_NAME}_", dir=args.workdir))
    experiments: dict[str, dict] = {}
    per_session_dump: dict[str, dict] = {}
    try:
        for experiment, spec in EXPERIMENTS.items():
            representation = spec["representation"]
            sessions = session_sets[spec["session_set"]]
            n_obs = int(sessions[0]["obs"].shape[0])
            sessions_dir = scratch / experiment
            write_sessions(sessions, pool, representation, sessions_dir)
            results, status, control_ranks = {}, {}, {}
            per_session_dump[experiment] = {}
            for name, command in solvers:
                output = scratch / f"{experiment}__{name}.json"
                log = scratch / f"{experiment}__{name}.log"
                error = run_solver(name, command, sessions_dir, output, log, args.timeout)
                scores = None
                if error is None:
                    scores, error = read_scores(output, sessions)
                if error is None:
                    means, per_session, control_mean = summarize(scores, sessions)
                    control_ranks[name] = control_mean
                    if args.n_control and control_mean < threshold:
                        error = (f"control sessions ranked at {control_mean:.2f} < {threshold:.2f}: "
                                 "ranking signal does not come from the EEG/stimulus pairing")
                if error is not None:
                    results[name] = failed_metrics(sessions)
                    status[name] = f"failed: {error}"
                    print(f"[{SLICE_NAME}] {experiment} {name}: {status[name]}", flush=True)
                    continue
                results[name] = means
                status[name] = "ok"
                per_session_dump[experiment][name] = per_session
            experiments[experiment] = {
                "slice": SLICE_NAME,
                "design": args.design,
                "eeg_representation": representation,
                "n_sessions": int(sum(not s["control"] for s in sessions)),
                "n_control_sessions": int(sum(s["control"] for s in sessions)),
                "n_observations": n_obs,
                "n_candidates": N_CANDIDATES,
                "results": results,
                "_status": status,
                "_control_mean_target_rank": control_ranks,
                "_control_gate_threshold": threshold,
            }
            shutil.rmtree(sessions_dir, ignore_errors=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    scores_out = {"_slice": SLICE_NAME, "_design": args.design, "experiments": experiments}
    path = Path(args.scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores_out, indent=2))
    if args.per_session_out:
        Path(args.per_session_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.per_session_out).write_text(json.dumps(per_session_dump))
    print(json.dumps(scores_out, indent=2))


if __name__ == "__main__":
    main()
