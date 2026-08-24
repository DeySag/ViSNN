"""Single tagged experiment: config dict -> subprocess training run.

    # CLI usage
    python ablation/run_experiment.py --track depth --seed 42 \
        --cfg synthetic=true --cfg epochs=2 \
        --cfg fire_fn=mtn --cfg lambda=0.25

    # Programmatic usage
    from ablation.run_experiment import build_tag, run_config
    tag = build_tag('depth', {'fire_fn': 'mtn', 'lambda': 0.25}, seed=42)
    result = run_config('depth', {...}, seed=42)

The config keys mirror the training CLIs' argparse destinations
(``epochs``, ``fire_fn``, ``lambda``, ``no_convert``, ...). Values are passed
through to ``train_depth.py`` / ``train_ssd.py`` verbatim; unknown keys are
rejected instead of being silently dropped, since a typo in an ablation grid
would otherwise quietly duplicate the baseline row.

Every run gets its own directory under the results root and writes
config.json / metrics.csv / summary.json / log.txt there. The runner is a
subprocess on purpose: identical behaviour to manual invocations, complete
isolation between experiments, and a log file even when the session dies.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    # Direct invocation puts ablation/ first on sys.path; config and friends
    # live one level up.
    sys.path.insert(0, PROJECT_ROOT)


def _default_seed():
    import config
    return config.SEED

TRACK_SCRIPTS = {
    'depth': os.path.join(PROJECT_ROOT, 'train_depth.py'),
    'ssd': os.path.join(PROJECT_ROOT, 'train_ssd.py'),
}

# store_true switches: appear in argv as bare flags when their value is True,
# are omitted entirely when False.
FLAG_KEYS = {
    'synthetic', 'no_pretrained', 'no_convert', 'search_lambda',
    'full_eval', 'eigen_crop', 'coco_metrics',
}

# Every argparse destination accepted across both tracks.
ALLOWED_KEYS = FLAG_KEYS | {
    'data_root', 'backbone', 'spatial_mode', 'output_activation',
    'epochs', 'batch_size', 'lr', 'weight_decay', 'alpha', 'gradient_mode',
    'num_workers', 'timesteps', 'fire_fn', 'lambda_', 'n_levels',
    'calibration_batches', 'percentile', 'crop_margin',
    'limit_train', 'limit_val', 'eval_batches', 'device', 'tag',
}
# 'lambda' reads nicer in configs than the argparse dest 'lambda_'.
ALLOWED_KEYS.add('lambda')

# Tag vocabulary: short, lowercase, filesystem-safe, ordered for readability.
# Only experimental variables contribute to the tag; operational knobs
# (EXCLUDED_FROM_TAG) live in config.json instead. The results root acts as
# the namespace, so keep synthetic test runs under a separate --run-root if
# their tags must not collide with real-data runs of the same grid point.
TAG_ORDER = [
    ('fire_fn', 'fn'), ('n_levels', 'L'), ('timesteps', 'T'),
    ('lambda', 'lam'), ('lambda_', 'lam'), ('percentile', 'p'),
    ('crop_margin', 'marg'), ('backbone', 'bb'), ('gradient_mode', 'grad'),
    ('alpha', 'a'), ('spatial_mode', 'sp'), ('epochs', 'ep'),
    ('limit_train', 'n'), ('calibration_batches', 'cb'), ('lr', 'lr'),
    ('synthetic', 'syn'), ('full_eval', 'fullev'),
    ('output_activation', 'act'),
]
FIRE_FN_ABBREV = {'binary': 'bin', 'mtn': 'mtn'}
GRADIENT_MODE_ABBREV = {'smoothness': 'smooth', 'matching': 'match'}
EXCLUDED_FROM_TAG = {
    'data_root', 'batch_size', 'num_workers', 'eval_batches', 'limit_val',
    'device', 'weight_decay', 'no_pretrained', 'search_lambda',
    'coco_metrics', 'tag',
}


def _format_value(value):
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, float):
        return f'{value:g}'
    value = str(value)
    return re.sub(r'[^A-Za-z0-9.\-]+', '', value)


def build_tag(track, cfg, seed):
    """Deterministic tag, e.g. depth_mtn_L8_lam0.25_seed42.

    Only keys explicitly present in ``cfg`` contribute, so a tag always names
    exactly what distinguishes this row from the defaults.
    """
    parts = [track]

    if cfg.get('no_convert'):
        parts.append('cont')

    consumed = {'no_convert'}
    for key, abbrev in TAG_ORDER:
        if key not in cfg:
            continue
        consumed.add(key)
        value = cfg[key]
        if key == 'fire_fn':
            value = FIRE_FN_ABBREV.get(str(value), str(value))
        elif key == 'gradient_mode':
            value = GRADIENT_MODE_ABBREV.get(str(value), str(value))
        parts.append(f'{abbrev}{_format_value(value)}')

    # Anything unrecognised but explicitly set still deserves to be in the tag
    # (operational knobs are skipped -- see EXCLUDED_FROM_TAG).
    for key in sorted(set(cfg) - consumed - {'seed'} - EXCLUDED_FROM_TAG):
        parts.append(f'{_format_value(key)}{_format_value(cfg[key])}')

    parts.append(f'seed{int(seed)}')
    return '_'.join(parts)


def cfg_to_argv(cfg):
    """Config dict -> CLI arguments of the training scripts."""
    argv = []
    for key, value in cfg.items():
        cli_key = 'lambda' if key in ('lambda', 'lambda_') else key
        if cli_key not in ALLOWED_KEYS:
            raise ValueError(
                f'Unknown config key {key!r}. Allowed: '
                f'{sorted(ALLOWED_KEYS)}')
        flag = '--' + cli_key.replace('_', '-')
        if cli_key in FLAG_KEYS:
            if value:
                argv.append(flag)
        else:
            if isinstance(value, bool):
                raise ValueError(
                    f'{key} takes a value, got boolean {value!r}')
            argv.extend([flag, str(value)])
    return argv


def _parse_scalar(text):
    """'0.25' -> float, 'true' -> True, 'mtn' -> 'mtn'."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def parse_cfg_pairs(pairs):
    """['key=value', ...] -> dict, with JSON-typed values."""
    cfg = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f'--cfg expects key=value, got {pair!r}')
        key, _, value = pair.partition('=')
        cfg[key.strip()] = _parse_scalar(value.strip())
    return cfg


def run_config(track, cfg, seed=None, run_root=None,
               timeout_seconds=None, dry_run=False, echo=True):
    """Execute one tagged run; returns a result dict.

    Skips nothing -- that decision belongs to the matrix runner, which knows
    about resume semantics. Here a completed marker simply overwrites.
    ``timeout_seconds`` is a hard watchdog: the training process is killed
    when it expires.
    """
    import config  # local import keeps module import cheap and side-effect free

    if track not in TRACK_SCRIPTS:
        raise ValueError(f'track must be one of {sorted(TRACK_SCRIPTS)}')
    seed = config.SEED if seed is None else seed

    tag = cfg.get('tag') or build_tag(track, cfg, seed)
    run_root = run_root or config.RESULTS_DIR
    run_dir = os.path.join(run_root, tag)

    argv = [
        sys.executable,
        TRACK_SCRIPTS[track],
        '--run-dir', run_dir,
        '--tag', tag,
        *cfg_to_argv(cfg),
    ]
    summary_path = os.path.join(run_dir, 'summary.json')
    if dry_run:
        print(' '.join(argv))
        return {'tag': tag, 'run_dir': run_dir, 'exit_code': None,
                'seconds': None, 'summary_path': summary_path}

    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [PROJECT_ROOT] + env.get('PYTHONPATH', '').split(os.pathsep)
    ).rstrip(os.pathsep)

    start = time.perf_counter()
    exit_code = None
    with open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8') as log:
        log.write(f'\n===== launch {time.strftime("%Y-%m-%d %H:%M:%S")} '
                  f'seed={seed} =====\n')
        log.flush()
        proc = subprocess.Popen(argv, cwd=PROJECT_ROOT, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace')

        watchdog = None
        if timeout_seconds:
            watchdog = threading.Timer(
                timeout_seconds, _kill_tree, args=(proc,))
            watchdog.daemon = True
            watchdog.start()
        try:
            for line in proc.stdout:
                log.write(line)
                if echo:
                    sys.stdout.write(line)
            exit_code = proc.wait()
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if proc.poll() is None:  # stream ended but process alive
                _kill_tree(proc)
                exit_code = proc.wait()
    seconds = time.perf_counter() - start

    return {
        'tag': tag,
        'run_dir': run_dir,
        'exit_code': exit_code,
        'seconds': seconds,
        'summary_path': summary_path if os.path.exists(summary_path) else None,
    }


def _kill_tree(proc):
    """Terminate a subprocess and its children (Windows-safe)."""
    try:
        proc.kill()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Run one tagged ablation experiment')
    parser.add_argument('--track', required=True, choices=sorted(TRACK_SCRIPTS))
    parser.add_argument('--cfg', action='append', default=[],
                        metavar='KEY=VALUE',
                        help='training argument; repeatable')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--run-root', default=None,
                        help='results root (default: config.RESULTS_DIR)')
    parser.add_argument('--timeout-minutes', type=float, default=None)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the command without executing it')
    args = parser.parse_args()

    cfg = parse_cfg_pairs(args.cfg)
    seed = args.seed if args.seed is not None else _default_seed()

    print(f'[run_experiment] track={args.track} seed={seed}')
    for key in sorted(cfg):
        print(f'  {key} = {cfg[key]}')

    timeout = (args.timeout_minutes * 60.0) if args.timeout_minutes else None
    result = run_config(args.track, cfg, seed=seed, run_root=args.run_root,
                        timeout_seconds=timeout, dry_run=args.dry_run)

    if not args.dry_run:
        status = 'done' if result['exit_code'] == 0 else (
            f'FAILED (exit {result["exit_code"]})')
        print(f'\n[run_experiment] {result["tag"]}: {status} '
              f'in {result["seconds"]:.1f}s')
        if not result['summary_path']:
            print('  warning: no summary.json written; the run did not finish')
    return result


if __name__ == '__main__':
    main()
