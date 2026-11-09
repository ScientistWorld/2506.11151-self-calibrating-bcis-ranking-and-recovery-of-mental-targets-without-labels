"""Aggregate completed evaluations; never used to prepare any method artifact."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / 'redteam'
reference = json.loads((ROOT / 'scoring/reference.json').read_text())['experiments']
cases = {}
errors = {}
for folder in sorted(RT.iterdir()):
    if not folder.is_dir() or not (folder/'status.json').exists():
        continue
    status = json.loads((folder/'status.json').read_text())
    if status['test_returncode'] != 0:
        errors[folder.name] = status
        continue
    doc = json.loads((folder/'scores_test.json').read_text())
    if any(not math.isfinite(float(v)) for e in doc['experiments'].values() for v in e['results']['lr'].values()):
        errors[folder.name] = {'reason': 'test evaluator emitted non-finite metrics; no valid rank'}
        continue
    cases[folder.name] = doc
anchor = cases['reference_seed0']
# Attempt-dependent normalization is unknowable before future attempts exist.
# Report a precisely defined audit-cohort normalization, plus all raw metrics.
scales = {}
for experiment, contract in reference.items():
    scales[experiment] = {}
    for metric in contract['metrics']:
        values = [case['experiments'][experiment]['results']['lr'][metric] for case in cases.values()]
        sd = float(np.std(values))
        scales[experiment][metric] = sd if sd > 0 else 1.0


def rank(case, baseline=anchor, scales=scales):
    total = 0.0
    contributions = {}
    for experiment, contract in reference.items():
        contributions[experiment] = {}
        result = case['experiments'][experiment]['results']['lr']
        base = baseline['experiments'][experiment]['results']['lr']
        for metric, spec in contract['metrics'].items():
            direction = 1 if spec['higher_is_better'] else -1
            contribution = contract['weight'] * abs(spec['coefficient']) * direction * (result[metric]-base[metric]) / scales[experiment][metric]
            contributions[experiment][metric] = contribution
            total += contribution
    return float(total), contributions

summary = {'normalization': 'population standard deviation across all completed finite lr audit attempts, including 3 reference seeds; zero std uses 1',
           'anchor': 'reference_seed0; reproduced scoring/scores_test.json, not paper-scale reference values',
           'coefficient_rule': 'abs(coefficient); direction only from higher_is_better; experiment weights 0.65/0.35',
           'scales': scales, 'cases': {}, 'invalid_or_failed': errors}
for name, case in cases.items():
    score, contributions = rank(case)
    cmp = []
    for experiment, contract in reference.items():
        result = case['experiments'][experiment]['results']['lr']
        base = anchor['experiments'][experiment]['results']['lr']
        cmp.extend((1 if spec['higher_is_better'] else -1)*(result[metric]-base[metric]) for metric,spec in contract['metrics'].items())
    summary['cases'][name] = {'normalized_gain': score,
                            'pareto': 'equal' if all(abs(x)<1e-9 for x in cmp) else ('better' if all(x>=-1e-9 for x in cmp) else ('worse' if all(x<=1e-9 for x in cmp) else 'mixed')),
                            'contributions': contributions,
                            'results': {e:b['results']['lr'] for e,b in case['experiments'].items()}}
seed_scores = [summary['cases'][f'reference_seed{i}']['normalized_gain'] for i in range(3)]
summary['reference_seed_spread'] = {'scores': seed_scores, 'max_minus_min': max(seed_scores)-min(seed_scores),
                                    'metric_spreads': {e:{m:max(cases[f'reference_seed{i}']['experiments'][e]['results']['lr'][m] for i in range(3))-min(cases[f'reference_seed{i}']['experiments'][e]['results']['lr'][m] for i in range(3)) for m in contract['metrics']} for e,contract in reference.items()}}
# Honest-only normalization makes the seed comparison independent of shortcuts.
honest_names = ['reference_seed0','reference_seed1','reference_seed2','better_more_data','weaker_shuffled']
honest_scales = {e:{m:float(np.std([cases[n]['experiments'][e]['results']['lr'][m] for n in honest_names])) or 1.0 for m in contract['metrics']} for e,contract in reference.items()}
summary['honest_only'] = {'names':honest_names, 'scales':honest_scales,
                          'gains':{n:rank(cases[n],scales=honest_scales)[0] for n in honest_names}}
h = summary['honest_only']['gains']
summary['honest_only']['seed_spread'] = max(h[f'reference_seed{i}'] for i in range(3))-min(h[f'reference_seed{i}'] for i in range(3))
# Baselines are ranked under their own lr slot for a direct comparison.
for control in ['dummy','s_lr']:
    case = json.loads(json.dumps(anchor))
    for e in reference:
        case['experiments'][e]['results']['lr'] = case['experiments'][e]['results'][control]
    summary.setdefault('baseline_gains',{})[control] = rank(case)[0]
(RT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({n:{'gain':v['normalized_gain'],'pareto':v['pareto']} for n,v in summary['cases'].items()},indent=2))
print('SEED_SPREAD', summary['reference_seed_spread']['max_minus_min'])
print('HONEST_ONLY', summary['honest_only']['gains'], 'spread', summary['honest_only']['seed_spread'])
