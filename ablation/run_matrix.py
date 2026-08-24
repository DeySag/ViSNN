"""Experiment matrix runner: ordered plans x seeds, resumable across sessions.

    # The headline table (Phase 2), depth track, 3 seeds
    python ablation/run_matrix.py --preset phase2_depth

    # A custom plan expressed as JSON
    python ablation/run_matrix.py --plan my_plan.json

    # Just show what would run
    python ablation/run_matrix.py --preset A1_lambda_depth --dry-run

A plan resolves to a flat, ordered list of (tag, track, cfg) entries. Before
each entry the runner checks ``results/<tag>/summary.json``; if it exists the
entry is logged as `skipped`, which is what makes interrupted Kaggle/Colab
sessions resumable: rerun the same command and only unfinished work executes.

Progress is appended to ``results/progress.log`` as
``<timestamp> | <tag> | <status> | <elapsed>s | <note>``
so a crashed session still leaves an audit trail.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation.run_experiment import (  # noqa: E402
    TRACK_SCRIPTS,
    build_tag,
    parse_cfg_pairs,
    run_config,
)

DEFAULT_SEEDS = [42, 43, 44]

# ---------------------------------------------------------------------------
# Presets. 'common' is merged into every experiment's cfg (experiments win).
# Headline presets intentionally set NO data flags: they assume real datasets.
# Add {"synthetic": true} or limit_train via --cfg overrides for local tests.
# ---------------------------------------------------------------------------
PRESETS = {
    # ---- Phase 2: headline table -----------------------------------------
    'phase2_depth': {
        'track': 'depth',
        'seeds': DEFAULT_SEEDS,
        'common': {'full_eval': True},
        'experiments': [
            {'no_convert': True},
            {'fire_fn': 'binary', 'lambda': 1.0},
            {'fire_fn': 'mtn', 'lambda': 1.0, 'n_levels': 8},
            {'fire_fn': 'mtn', 'n_levels': 8, 'search_lambda': True},
        ],
    },
    'phase2_ssd': {
        'track': 'ssd',
        'seeds': DEFAULT_SEEDS,
        'common': {'full_eval': True, 'coco_metrics': True},
        'experiments': [
            {'no_convert': True},
            {'fire_fn': 'binary', 'lambda': 1.0},
            {'fire_fn': 'mtn', 'lambda': 1.0, 'n_levels': 8},
            {'fire_fn': 'mtn', 'n_levels': 8, 'search_lambda': True},
        ],
    },

    # ---- Phase 3 ablations (1 seed, fixed 8k-frame subset per plan) -------
    # A1: lambda sweep -> accuracy-vs-spike-rate Pareto (core figure).
    'A1_lambda_depth': {
        'track': 'depth',
        'seeds': [42],
        'common': {'fire_fn': 'mtn', 'n_levels': 8, 'limit_train': 8000,
                   'epochs': 30},
        'experiments': [
            {'lambda': lam} for lam in (0.1, 0.25, 0.5, 0.75, 1.0)
        ],
    },
    # A2: timesteps sweep -> rate-coded convergence vs latency/energy cost.
    'A2_timesteps_depth': {
        'track': 'depth',
        'seeds': [42],
        'common': {'fire_fn': 'mtn', 'n_levels': 8, 'lambda': 0.25,
                   'limit_train': 8000, 'epochs': 30},
        'experiments': [{'timesteps': t} for t in (1, 2, 4, 8, 16)],
    },
    # A3: MTN levels sweep -> bits-vs-accuracy curve.
    'A3_levels_depth': {
        'track': 'depth',
        'seeds': [42],
        'common': {'fire_fn': 'mtn', 'lambda': 0.25, 'limit_train': 8000,
                   'epochs': 30},
        'experiments': [{'n_levels': n} for n in (2, 4, 8, 16)],
    },
    # B-A1: detection lambda sweep (the only detection ablation in budget).
    'B_A1_lambda_ssd': {
        'track': 'ssd',
        'seeds': [42],
        'common': {'fire_fn': 'mtn', 'n_levels': 8, 'limit_train': 50000,
                   'limit_val': 5000, 'epochs': 12, 'batch_size': 16},
        'experiments': [
            {'lambda': lam} for lam in (0.1, 0.25, 0.5, 1.0)
        ],
    },
}


def resolve_plan(spec, seeds_override=None):
    """Preset name / path / dict -> ordered list of dicts:
    [{'tag', 'track', 'cfg', 'seed'}, ...]"""
    if isinstance(spec, str) and spec in PRESETS:
        spec = PRESETS[spec]
    elif isinstance(spec, str) and os.path.exists(spec):
        with open(spec, encoding='utf-8') as handle:
            spec = json.load(handle)
    elif not isinstance(spec, dict):
        raise ValueError(
            f'Unknown plan {spec!r}; pass a preset name '
            f'({sorted(PRESETS)}) or a JSON file')

    track = spec.get('track')
    if track not in TRACK_SCRIPTS:
        raise ValueError(f'plan needs a valid "track", got {track!r}')
    seeds = seeds_override or spec.get('seeds') or DEFAULT_SEEDS
    common = dict(spec.get('common') or {})
    experiments = spec.get('experiments')
    if not experiments:
        raise ValueError('plan has no experiments')

    entries = []
    seen_tags = set()
    for exp in experiments:
        cfg = {**common, **exp}
        for seed in seeds:
            tag = build_tag(track, cfg, seed)
            if tag in seen_tags:      # identical cfg+seed rows collapse
                continue
            seen_tags.add(tag)
            entries.append({'tag': tag, 'track': track,
                            'cfg': cfg, 'seed': seed})
    return entries


class ProgressLog:
    """Append-only audit trail shared by every session."""

    def __init__(self, path):
        self.path = path

    def write(self, tag, status, seconds=None, note=''):
        elapsed = f'{seconds:.1f}s' if seconds else '-'
        line = (f'{datetime.now().isoformat(timespec="seconds")} | {tag} | '
                f'{status:<8} | {elapsed:>10} | {note}')
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')
        print(f'[matrix] {line}')


def run_matrix(entries, run_root=None, timeout_minutes=None, dry_run=False,
               stop_on_fail=False, max_runs=None, progress_path=None):
    import config

    run_root = run_root or config.RESULTS_DIR
    progress = ProgressLog(progress_path or
                           os.path.join(run_root, 'progress.log'))
    timeout_seconds = timeout_minutes * 60.0 if timeout_minutes else None

    executed = skipped = failed = 0
    for i, entry in enumerate(entries, start=1):
        tag = entry['tag']
        summary_path = os.path.join(run_root, tag, 'summary.json')
        header = (f'[{i}/{len(entries)}] {tag}')

        if os.path.exists(summary_path):
            progress.write(tag, 'skipped', note='summary.json already exists')
            skipped += 1
            continue
        if max_runs is not None and executed >= max_runs:
            print(f'{header}: budget reached (--max-runs {max_runs}), stopping')
            break

        if dry_run:
            progress.write(tag, 'planned', note='dry run')
            continue

        print(f'\n=== {header} ===')
        result = run_config(entry['track'], entry['cfg'], seed=entry['seed'],
                            run_root=run_root, timeout_seconds=timeout_seconds)
        if result['exit_code'] == 0 and result['summary_path']:
            progress.write(tag, 'done', seconds=result['seconds'])
            executed += 1
        else:
            progress.write(tag, 'failed', seconds=result['seconds'],
                           note=f'exit={result["exit_code"]}')
            failed += 1
            if stop_on_fail:
                print('[matrix] --stop-on-fail set, aborting')
                break

    print(f'\n[matrix] summary: {executed} run(s), {skipped} skipped, '
          f'{failed} failed, {len(entries)} total')
    return {'executed': executed, 'skipped': skipped, 'failed': failed}


def main():
    parser = argparse.ArgumentParser(description='Run an experiment matrix')
    parser.add_argument('--preset', choices=sorted(PRESETS),
                        help='built-in experiment plan')
    parser.add_argument('--plan', help='path to a JSON plan file')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='override the plan seeds')
    parser.add_argument('--cfg', action='append', default=[],
                        metavar='KEY=VALUE',
                        help='extra cfg merged into EVERY experiment '
                             '(e.g. --cfg synthetic=true)')
    parser.add_argument('--run-root', default=None)
    parser.add_argument('--timeout-minutes', type=float, default=None)
    parser.add_argument('--max-runs', type=int, default=None,
                        help='execute at most this many runs this session')
    parser.add_argument('--stop-on-fail', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='list planned runs without executing them')
    args = parser.parse_args()

    if not args.preset and not args.plan:
        parser.error('pass --preset <name> or --plan <file.json>')

    entries = resolve_plan(args.preset or args.plan, seeds_override=args.seeds)

    extra = parse_cfg_pairs(args.cfg)
    if extra:
        rebuilt = []
        for e in entries:
            cfg = {**e['cfg'], **extra}
            tag = build_tag(e['track'], cfg, e['seed'])
            if any(r['tag'] == tag for r in rebuilt):
                continue
            rebuilt.append({**e, 'cfg': cfg, 'tag': tag})
        entries = rebuilt

    import config
    config.ensure_dirs()
    run_root = args.run_root or config.RESULTS_DIR
    os.makedirs(run_root, exist_ok=True)

    if args.dry_run:
        print(f'[matrix] {len(entries)} entr(ies):')
        for e in entries:
            done = os.path.exists(os.path.join(run_root, e['tag'],
                                               'summary.json'))
            state = 'DONE' if done else 'todo '
            print(f'  [{state}] {e["tag"]}  ({e["track"]}, seed {e["seed"]})')
        return

    snapshot = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'entries': [{'tag': e['tag'], 'track': e['track'],
                     'seed': e['seed'], 'cfg': e['cfg']} for e in entries],
    }
    snap_name = ('matrix_' +
                 (args.preset or os.path.basename(args.plan).split('.')[0]) +
                 '.json')
    with open(os.path.join(run_root, snap_name), 'w', encoding='utf-8') as h:
        json.dump(snapshot, h, indent=2)

    run_matrix(entries, run_root=args.run_root,
               timeout_minutes=args.timeout_minutes, dry_run=False,
               stop_on_fail=args.stop_on_fail, max_runs=args.max_runs)


if __name__ == '__main__':
    main()
