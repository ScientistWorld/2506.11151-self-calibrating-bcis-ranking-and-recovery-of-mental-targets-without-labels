"""Shortcut solvers replayed against the hardened evaluator.

Each kind is an ordinary solver: it reads an evaluator-written session
directory and writes {"scores": {session_id: [...]}}.  Only `hidden_oracle`
reads evaluator code (to show that the control-session gate catches a solver
whose information does not come from the EEG/stimulus pairing); the others use
only what any submission can see.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from data.cursor_data import load_arrays, read_session_dir, unique_targets  # noqa: E402


def _public_pool():
    arrays = load_arrays()
    recorded = np.linalg.norm(arrays["observed_faces"] - arrays["target_faces"], axis=1)
    return arrays, recorded


def _lookup_scores(eeg, stimuli, candidates, public_eeg, recorded):
    """Label lookup: match each session EEG row to its nearest public row and
    read off that row's recorded stimulus-target distance."""
    pub = (public_eeg - public_eeg.mean(0)) / np.where(public_eeg.std(0) > 0, public_eeg.std(0), 1)
    sess = (eeg - eeg.mean(0)) / np.where(eeg.std(0) > 0, eeg.std(0), 1)
    pub_sq = np.sum(pub**2, axis=1)
    matched = np.empty(len(sess), dtype=int)
    for start in range(0, len(sess), 500):
        block = sess[start:start + 500]
        d2 = pub_sq[None, :] - 2 * block @ pub.T
        matched[start:start + 500] = np.argmin(d2, axis=1)
    decoded = recorded[matched]
    dist = cdist(stimuli, candidates)
    dist = (dist - dist.mean(0)) / dist.std(0)
    decoded = (decoded - decoded.mean()) / max(decoded.std(), 1e-12)
    return dist.T @ decoded / len(decoded)


def _hidden_oracle(session_ids, candidates_by_id):
    spec = importlib.util.spec_from_file_location("hidden_eval", WORKSPACE / "eval/test/evaluate_cursor.py")
    hidden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hidden)
    pool = hidden.load_pool(WORKSPACE / "data/downloads/all_data_sorted.npz")
    sessions = hidden.make_sessions(pool, hidden.DEFAULT_DESIGN, hidden.SPLIT_SEED,
                                    hidden.N_ALIGNED, hidden.N_CONTROL)
    sessions += hidden.make_sessions(pool, hidden.DEFAULT_DESIGN, hidden.SPLIT_SEED + 1,
                                     hidden.N_ALIGNED, hidden.N_CONTROL, hidden.LOW_DATA_OBSERVATIONS)
    truth = {s["id"]: s["true_distances"] for s in sessions}
    return {sid: -truth[sid] for sid in session_ids if sid in truth}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=[
        "centroid", "centrality", "density", "catalogue", "eeg_lookup", "constant", "nan", "missing",
        "scorefile_forgery", "hidden_oracle"])
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sessions = list(read_session_dir(args.sessions))
    scores = {}
    if args.kind == "hidden_oracle":
        oracle = _hidden_oracle([sid for sid, _ in sessions], {sid: s["candidates"] for sid, s in sessions})
        for sid, session in sessions:
            scores[sid] = oracle.get(sid, np.zeros(len(session["candidates"])))
    else:
        if args.kind in {"catalogue", "eeg_lookup"}:
            arrays, recorded = _public_pool()
            catalogue = unique_targets(arrays["target_faces"])
            index = json.loads((Path(args.sessions) / "index.json").read_text())
            public_eeg = np.asarray(arrays[index["eeg_representation"]], dtype=float)
        for n, (sid, session) in enumerate(sessions):
            stimuli = session["stimuli"].astype(float)
            candidates = session["candidates"].astype(float)
            if args.kind == "centroid":
                values = -np.linalg.norm(candidates - stimuli.mean(axis=0), axis=1)
            elif args.kind == "centrality":
                values = -cdist(candidates, candidates).mean(axis=1)
            elif args.kind == "density":
                d = cdist(stimuli, candidates)
                values = -np.sort(d, axis=0)[:20].mean(axis=0)
            elif args.kind == "catalogue":
                values = -np.min(cdist(candidates, catalogue), axis=1)
            elif args.kind == "eeg_lookup":
                values = _lookup_scores(session["eeg"].astype(float), stimuli, candidates,
                                        public_eeg, recorded)
            elif args.kind in {"constant", "scorefile_forgery"}:
                values = np.zeros(len(candidates))
            elif args.kind == "nan":
                values = np.full(len(candidates), np.nan)
            elif args.kind == "missing":
                if n % 2:
                    continue
                values = np.zeros(len(candidates))
            scores[sid] = values

    if args.kind == "scorefile_forgery":
        # Plant perfect scores at the path the evaluator will write; the
        # evaluator must overwrite them with its own recomputed metrics.
        forged = {"forged": True, "experiments": {"unbiased_ranking_raw_eeg": {"results": {
            "scorefile_forgery": {"target_rank": 1.0, "top_rank_distance": 0.0}}}}}
        planted = Path(os.environ.get("FORGE_PATH", WORKSPACE / "scoring/scores_test.json"))
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text(json.dumps(forged))
    payload = {"scores": {sid: [float(v) for v in values] for sid, values in scores.items()}}
    Path(args.output).write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
