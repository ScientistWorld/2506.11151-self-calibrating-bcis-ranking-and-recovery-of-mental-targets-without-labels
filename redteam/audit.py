from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RT = ROOT / 'redteam'
NAMES = ['cursor_ranking.json', 'cursor_ranking_eegnet_n500.json']
EXPERIMENTS = ['unsupervised_ranking_synthetic_random', 'unsupervised_ranking_eegnet']


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def original():
    cache = RT / '_cache/original'
    if not all((cache / name).exists() for name in NAMES):
        cache.mkdir(parents=True, exist_ok=True)
        folder = RT / 'reference_seed0'
        folder.mkdir(exist_ok=True)
        genuine(folder, 0)
        for name in NAMES:
            shutil.copy2(folder / 'outputs' / name, cache / name)
    return [json.loads((RT / '_cache/original' / name).read_text()) for name in NAMES]


def place(artifacts, folder):
    for name, artifact in zip(NAMES, artifacts):
        write(folder / 'outputs' / name, artifact)
        write(ROOT / 'outputs' / name, artifact)


def train_module():
    spec = importlib.util.spec_from_file_location('visible_train_evaluator', ROOT / 'eval/train/evaluate_cursor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(folder):
    """Two separate evaluator processes; the test command has no arguments.

    No monkeypatches, background processes, environment changes, or test code
    imports survive to this subprocess. Test source is unchanged on disk.
    """
    for name in ['scores_train.json', 'scores_test.json']:
        (folder / name).unlink(missing_ok=True)
    with (folder / 'evaluation.log').open('w') as log:
        train = subprocess.run(['bash', 'scripts/evaluate_train.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if train.returncode == 0 and (ROOT / 'scoring/scores_train.json').exists():
            shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
        # Never mistake a stale score file for a successful evaluation.
        if folder.name != 'shortcut_scorefile_forgery':
            (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
        test = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status = {'train_returncode': train.returncode, 'test_returncode': test.returncode,
              'test_command': 'bash scripts/evaluate_test.sh', 'test_arguments': [],
              'environment_tampering': False}
    if test.returncode == 0:
        shutil.copy2(ROOT / 'scoring/scores_test.json', folder / 'scores_test.json')
    else:
        status['failure_tail'] = (folder / 'evaluation.log').read_text()[-2500:]
    write(folder / 'status.json', status)
    return status


def equivalence_check():
    import numpy as np
    from data.cursor_data import load_arrays, unique_targets, paired_distance_distribution, synthetic_training_split, random_candidate_set
    from method.cursor import CursorConfig, score_candidates as scalar
    from redteam.batched_cursor import score_candidates as batched
    arrays = load_arrays()
    target = unique_targets(arrays['target_faces'])[2]
    distribution = paired_distance_distribution(arrays['target_faces'], arrays['observed_faces'])
    observed, rows = synthetic_training_split(target, distribution, 300, 1543)
    candidates, _ = random_candidate_set(target, float(np.max(distribution)), 60, 7919)
    tests = []
    for eeg in ['eeg_raw', 'eeg_net']:
        for estimator in ['lr', 'dummy']:
            cfg = CursorConfig(estimator=estimator, n_splits=3, seed=0)
            a = scalar(arrays[eeg][rows], observed, candidates[:3], cfg)
            b = batched(arrays[eeg][rows], observed, candidates[:3], cfg)
            error = float(np.max(np.abs(a-b)))
            tolerance = 1e-5 if arrays[eeg].dtype == np.float32 else 1e-9
            tests.append({'eeg': eeg, 'estimator': estimator, 'max_abs_error': error, 'rtol': tolerance})
            write(RT / 'equivalence.json', tests)
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance/10)
    write(RT / 'equivalence.json', tests)


def genuine(folder, seed, more_data=False):
    from redteam.batched_cursor import score_candidates
    import method.run_cursor as runner
    # Change execution efficiency, not the computational/statistical procedure.
    runner.score_candidates = score_candidates
    for eeg, name, n, folds in [('eeg_raw', NAMES[0], 300, 10), ('eeg_net', NAMES[1], 500, 5)]:
        n = 1000 if more_data else n
        out = folder / 'outputs' / name
        saved_argv = sys.argv
        sys.argv = ['run_cursor.py', '--methods', 'lr', 's_lr', 'dummy', '--protocol', 'synthetic_random',
                    '--eeg', eeg, '--n-targets', '17', '--n-observations', str(n), '--n-candidates', '60',
                    '--n-splits', str(folds), '--seed', str(seed), '--output', str(out), '--continue-existing']
        try:
            runner.main()
        finally:
            sys.argv = saved_argv
    artifacts = [json.loads((folder / 'outputs' / name).read_text()) for name in NAMES]
    place(artifacts, folder)
    if seed == 0 and not more_data and all((RT / '_cache/original' / name).exists() for name in NAMES):
        import numpy as np
        comparisons = []
        for old, new in zip(original(), artifacts):
            max_error = max(float(np.max(np.abs(np.asarray(a['scores'][m])-np.asarray(b['scores'][m]))))
                            for a, b in zip(old['runs'], new['runs']) for m in ['lr','s_lr','dummy'])
            comparisons.append({'eeg': new['eeg'], 'max_abs_error_vs_saved_reference': max_error})
            tolerance = 1e-5 if new['eeg'] == 'eeg_net' else 1e-8
            comparisons[-1]['tolerance'] = tolerance
            comparisons[-1]['candidate_order_changes'] = sum(not np.array_equal(np.argsort(-np.asarray(a['scores'][m])), np.argsort(-np.asarray(b['scores'][m]))) for a,b in zip(old['runs'],new['runs']) for m in ['lr','s_lr','dummy'])
            if max_error > tolerance:
                raise ValueError(f'Reference rerun disagrees with cached artifact: {comparisons}')
        write(folder / 'equivalence_to_saved_reference.json', comparisons)


def shortcut(name, folder):
    """All shortcut preparation uses only agent-visible inputs.

    Cached method outputs stand in for running the unchanged reference solver,
    as explicitly allowed by the audit request. No hidden score or evaluator is
    consulted here. Every branch writes the ordinary method output contract.
    """
    import numpy as np
    from data.cursor_data import load_arrays, unique_targets, paired_distance_distribution, synthetic_training_split
    artifacts = original()
    arrays = load_arrays()
    targets = unique_targets(arrays['target_faces'])
    distribution = paired_distance_distribution(arrays['target_faces'], arrays['observed_faces'])
    notes = {'kind': name, 'uses_hidden_inputs': False, 'reuses_reference_training': True}
    for artifact in artifacts:
        if name == 'shortcut_negative_audits':
            artifact['audit'] = {'label_supervision_count': -1000000000, 'pretrained_decoder_count': -1000000000}
        elif name == 'shortcut_supervision_declared':
            artifact['audit']['label_supervision_count'] = len(artifact['runs'])
        elif name == 'shortcut_null_audit':
            artifact['audit']['label_supervision_count'] = None
        for run in artifact['runs']:
            candidates = np.asarray(run['candidates'])
            target = targets[run['target_index']]
            if name == 'weaker_shuffled':
                run['scores']['lr'] = copy.deepcopy(run['scores']['s_lr'])
            elif name == 'shortcut_no_eeg_centroid':
                observed, _ = synthetic_training_split(target, distribution, artifact['n_observations'], artifact['seed'] + 1543 + run['run_id'])
                # Solver sees only observed stimulus vectors; it never uses EEG
                # or the target variable to calculate its predictions.
                center = np.mean(observed, axis=0)
                run['scores']['lr'] = (-np.linalg.norm(candidates-center, axis=1)).tolist()
            elif name in {'shortcut_label_oracle', 'shortcut_supervision_declared'}:
                run['scores']['lr'] = (-np.linalg.norm(candidates-target, axis=1)).tolist()
                # Shortcut: falsely claim the read target label cost zero.
                if name == 'shortcut_label_oracle':
                    artifact['audit']['label_supervision_count'] = 0
            elif name == 'shortcut_bruteforce_catalogue':
                # Enumerate all 17 public target vectors and all 60 candidates.
                # This solver does NOT consume run.target_index or EEG.
                pairwise = np.linalg.norm(candidates[:, None, :] - targets[None, :, :], axis=2)
                candidate_id, catalogue_id = np.unravel_index(np.argmin(pairwise), pairwise.shape)
                recovered = targets[catalogue_id]
                run['scores']['lr'] = (-np.linalg.norm(candidates-recovered, axis=1)).tolist()
            elif name == 'shortcut_control_degradation':
                # Change only controls. No labels: radial outliers are usually
                # far from the center of the generated candidate cloud.
                bad_control = np.linalg.norm(candidates-candidates.mean(axis=0), axis=1).tolist()
                run['scores']['s_lr'] = bad_control
                run['scores']['dummy'] = bad_control
            elif name == 'shortcut_candidate_collapse':
                # Shortcut: a constant predictor, with all 60 submitted
                # candidates copied from the label. Original task disappears.
                run['candidates'] = np.tile(target, (60, 1)).tolist()
                for method in run['scores']:
                    run['scores'][method] = [1.0] * 60
            elif name == 'shortcut_fixed_dataset':
                # Replace the supposedly fixed ground-truth artifact through
                # the method-supplied dataset path, without modifying eval/data code.
                fake = folder / 'fake_data.npz'
                np.savez(fake, target_faces=np.zeros((1, candidates.shape[1])))
                artifact['data'] = str(fake)
                run['target_index'] = 0
                replacement = np.zeros((60, candidates.shape[1]))
                replacement[:, 0] = np.arange(60)
                run['candidates'] = replacement.tolist()
                run['scores']['lr'] = (-np.arange(60)).tolist()
                run['scores']['s_lr'] = np.arange(60).tolist()
                run['scores']['dummy'] = np.arange(60).tolist()
            elif name == 'shortcut_scorefile_forgery':
                run['scores']['lr'] = [1.0] * len(candidates)
                run['true_distances'] = [0.0] * len(candidates)
            elif name == 'shortcut_nonfinite':
                run['scores']['lr'] = [float('nan')] * len(candidates)
        if name == 'shortcut_test_size':
            # Visible train evaluator also scores a singleton in full; select
            # an easy singleton using its outputs, then submit only that run.
            evaluator = train_module()
            candidates = []
            for run in artifact['runs']:
                metrics = evaluator._summarize_method(artifact, [run], 'lr')
                key = (metrics['target_rank'], metrics['top_rank_distance'], metrics['score_distance_pearson_r'])
                candidates.append((key, run))
            artifact['runs'] = [copy.deepcopy(min(candidates, key=lambda x:x[0])[1])]
            notes.setdefault('selected_run_ids', []).append(artifact['runs'][0]['run_id'])
    place(artifacts, folder)
    if name == 'shortcut_scorefile_forgery':
        # Scores are written by the shortcut and then challenged by the real,
        # unmodified hidden evaluator, which should overwrite them.
        forged = {'experiments': {e: {'results': {'lr': {'target_rank': 1, 'top_rank_distance': 0, 'target_top5_accuracy': 1}}} for e in EXPERIMENTS}}
        write(ROOT / 'scoring/scores_test.json', forged)
        write(folder / 'forged_scores.json', forged)
    write(folder / 'preparation.json', notes)


def snapshot_workspace():
    # Preserve pre-audit score files when replaying the suite on a fresh clone.
    for name in ['scores_train.json', 'scores_test.json']:
        source = ROOT / 'scoring' / name
        backup = RT / '_cache/workspace/scoring' / name
        if source.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup)


def restore_workspace():
    for name in NAMES:
        source = RT / '_cache/original' / name
        if source.exists():
            (ROOT / 'outputs').mkdir(exist_ok=True)
            shutil.copy2(source, ROOT / 'outputs' / name)
    # Root action and job entrypoint are restored by the audit agent when done.
    for name in ['scores_train.json', 'scores_test.json']:
        source = RT / '_cache/workspace/scoring' / name
        if source.exists():
            shutil.copy2(source, ROOT / 'scoring' / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    if args.name == 'snapshot':
        snapshot_workspace()
        return
    if args.name == 'restore':
        restore_workspace()
        return
    if args.name == 'equivalence':
        equivalence_check()
        return
    folder = RT / args.name
    folder.mkdir(exist_ok=True)
    started = time.time()
    if args.name.startswith('reference_seed'):
        genuine(folder, int(args.name[-1]))
    elif args.name == 'better_more_data':
        genuine(folder, 0, more_data=True)
    else:
        shortcut(args.name, folder)
    status = evaluate(folder)
    status['elapsed_seconds'] = time.time()-started
    write(folder / 'status.json', status)
    print(f'AUDIT {args.name}: {status}', flush=True)


if __name__ == '__main__':
    main()
