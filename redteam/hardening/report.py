"""Summarise the hardening audit (pure Python; runs anywhere).

gain(attempt) = sum_exp weight * sum_metric |coefficient| * directed(metric - ref) / SD(metric)

with directed() taken in each metric's better direction, the reference being
CURSOR-LR seed 0 on the hidden test, and SD the population standard deviation
of the metric across the audit attempts that produced valid scores.  Paired
comparisons use per-session values: every solver sees identical sessions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def metric_specs():
    ref = json.loads((WORKSPACE / "scoring/reference.json").read_text())["experiments"]
    return {exp: {"weight": spec["weight"],
                  "metrics": {m: (abs(d["coefficient"]), 1.0 if d["higher_is_better"] else -1.0)
                              for m, d in spec["metrics"].items()}}
            for exp, spec in ref.items()}


def collect(results_dir: Path, design: str, split: str, groups=("reference", "variants", "shortcuts")):
    attempts, status, per_session, control = {}, {}, {}, {}
    for group in groups:
        scores = load(results_dir / design / f"{split}_{group}.json")
        sessions = load(results_dir / design / f"{split}_{group}_sessions.json") or {}
        if not scores:
            continue
        for exp, block in scores["experiments"].items():
            for name, metrics in block["results"].items():
                key = "lr" if name == "lr" else name
                attempts.setdefault(key, {})[exp] = metrics
                status.setdefault(key, {})[exp] = block["_status"][name]
                control.setdefault(key, {})[exp] = block.get("_control_mean_target_rank", {}).get(name)
                if name in sessions.get(exp, {}):
                    per_session.setdefault(key, {})[exp] = sessions[exp][name]
    return attempts, status, per_session, control


def gains(attempts, status, specs, reference="lr"):
    valid = [a for a in attempts if all(s == "ok" for s in status[a].values())]
    sds = {}
    for exp, spec in specs.items():
        for metric in spec["metrics"]:
            values = [attempts[a][exp][metric] for a in valid if exp in attempts[a]]
            sd = statistics.pstdev(values) if len(values) > 1 else 0.0
            sds[(exp, metric)] = sd if sd > 1e-12 else 1.0
    out = {}
    for name, by_exp in attempts.items():
        total = 0.0
        for exp, spec in specs.items():
            if exp not in by_exp or exp not in attempts[reference]:
                continue
            exp_sum = 0.0
            for metric, (coef, direction) in spec["metrics"].items():
                diff = by_exp[exp][metric] - attempts[reference][exp][metric]
                exp_sum += coef * direction * diff / sds[(exp, metric)]
            total += spec["weight"] * exp_sum
        out[name] = total
    return out, sds, valid


def paired(per_session, specs, sds, name, reference="lr", n_boot=4000, seed=0):
    """Per-session gain of `name` over the reference with a bootstrap 95% CI."""
    if name not in per_session or reference not in per_session:
        return None
    exps = [e for e in specs if e in per_session[name] and e in per_session[reference]]
    if not exps:
        return None
    n = len(next(iter(per_session[reference][exps[0]].values())))
    session_gain = [0.0] * n
    detail = {}
    for exp in exps:
        spec = specs[exp]
        for metric, (coef, direction) in spec["metrics"].items():
            a = per_session[name][exp][metric]
            b = per_session[reference][exp][metric]
            for i in range(n):
                session_gain[i] += spec["weight"] * coef * direction * (a[i] - b[i]) / sds[(exp, metric)]
            if metric in ("top_rank_distance", "target_rank", "score_distance_pearson_r"):
                diffs = [x - y for x, y in zip(a, b)]
                detail[f"{exp}.{metric}"] = {
                    "mean_diff": statistics.fmean(diffs),
                    "sessions_better": sum(direction * d > 0 for d in diffs),
                    "sessions_worse": sum(direction * d < 0 for d in diffs),
                }
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(session_gain, k=n)) for _ in range(n_boot))
    return {
        "mean_session_gain": statistics.fmean(session_gain),
        "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
        "sessions_positive": sum(g > 0 for g in session_gain),
        "sessions_negative": sum(g < 0 for g in session_gain),
        "n_sessions": n,
        "metrics": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(WORKSPACE / "redteam/hardening/results"))
    args = parser.parse_args()
    results = Path(args.results)
    specs = metric_specs()
    summary = {"formula": __doc__.strip().splitlines()[2], "designs": {}}

    for design_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        design = design_dir.name
        entry = {}
        for split in ("test", "train"):
            attempts, status, per_session, control = collect(results, design, split)
            if "lr" not in attempts:
                continue
            g, sds, valid = gains(attempts, status, specs)
            entry[split] = {
                "metrics": attempts,
                "status": status,
                "control_mean_target_rank": control,
                "gain_vs_reference": g,
                "valid_attempts_for_sd": valid,
                "sd": {f"{e}.{m}": v for (e, m), v in sds.items()},
                "paired_vs_reference": {name: paired(per_session, specs, sds, name)
                                        for name in attempts if name != "lr" and name in per_session},
            }
            seeds = [g[s] for s in ("lr", "lr_seed1", "lr_seed2") if s in g]
            if len(seeds) > 1:
                entry[split]["seed_gain_spread"] = max(seeds) - min(seeds)
        summary["designs"][design] = entry

    paper = load(results / "paper_protocol_check.json")
    if paper:
        summary["paper_protocol_check"] = paper["summary"]
    equivalence = load(WORKSPACE / "redteam/hardening/equivalence.json")
    if equivalence:
        summary["equivalence"] = equivalence
    (results / "summary.json").write_text(json.dumps(summary, indent=2))

    for design, entry in summary["designs"].items():
        for split, block in entry.items():
            print(f"\n=== {design} / {split} ===")
            for name in sorted(block["gain_vs_reference"], key=lambda n: -block["gain_vs_reference"][n]):
                row = []
                for exp, metrics in block["metrics"][name].items():
                    row.append(f"{exp.replace('unbiased_ranking_', '')}: rank {metrics['target_rank']:.2f} "
                               f"d {metrics['top_rank_distance']:.2f} r {metrics['score_distance_pearson_r']:.3f} "
                               f"top1 {metrics['target_top1_accuracy']:.3f}")
                p = block["paired_vs_reference"].get(name)
                ci = f" CI[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]" if p else ""
                state = "ok" if all(s == "ok" for s in block["status"][name].values()) else "FAILED"
                print(f"{name:28s} gain {block['gain_vs_reference'][name]:+7.3f}{ci} [{state}] | " + " | ".join(row))
            if "seed_gain_spread" in block:
                print(f"seed gain spread: {block['seed_gain_spread']:.3f}")


if __name__ == "__main__":
    main()
