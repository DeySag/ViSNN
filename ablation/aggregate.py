"""Aggregate run summaries into all_runs.csv and a RESULTS.md report.

    python ablation/aggregate.py                    # defaults: results/
    python ablation/aggregate.py --results-root results --out-md RESULTS.md

Reads every ``results/<tag>/summary.json`` (plus legacy flat
``results/<tag>_summary.json`` files), flattens them into one CSV row per run,
groups rows by tag with the ``_seed<N>`` suffix stripped, and emits tables of
mean +/- std over seeds.

The markdown is regenerated from scratch on every invocation; the CSV gains a
row per completed run. Both are cheap enough to run after every session, and
the CSV is the plotting-friendly source of truth for figures.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from datetime import datetime
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED_SUFFIX = re.compile(r'_seed\d+$')

# Row layout: identity, configuration, metrics, energy. Keys are emitted in
# this order when present; anything extra found in summaries is appended so no
# information silently disappears from the CSV.
CONFIG_KEYS = ['seed', 'fire_fn', 'lambda_', 'n_levels', 'timesteps',
               'percentile', 'crop_margin', 'backbone', 'gradient_mode',
               'alpha', 'spatial_mode', 'epochs', 'limit_train',
               'no_convert', 'synthetic', 'full_eval']
DEPTH_METRIC_KEYS = ['val_rmse', 'val_mae', 'val_abs_rel', 'val_sq_rel',
                     'val_rmse_log', 'val_delta1', 'val_delta2', 'val_delta3']
DET_METRIC_KEYS = ['mAP', 'mAP@[.5:.95]', 'num_classes_evaluated']
ENERGY_KEYS = ['spike_rate', 'spikes_per_frame', 'sops_per_frame',
               'ann_macs_per_frame', 'mac_equivalent_ratio',
               'ann_joules_per_frame', 'snn_joules_per_frame',
               'energy_ratio_ann_over_snn', 'latency_snn_T1_ms']


def flatten_summary(summary):
    """One summary.json -> flat row dict."""
    args = summary.get('args') or {}
    # 'track' lives at the summary top level (older flat summaries lack it;
    # the mAP heuristic below covers those).
    track = summary.get('track') or ('ssd' if 'best_val_mAP' in summary
                                     else 'depth')
    tag = args.get('tag')
    row = {'tag': tag,
           'group': SEED_SUFFIX.sub('', tag or ''),
           'track': track,
           'best_epoch': summary.get('best_epoch')}

    for key in CONFIG_KEYS:
        row[key] = args.get(key)

    # Unified headline metric regardless of track.
    if 'best_val_rmse' in summary:
        row['best_val_rmse'] = summary['best_val_rmse']
        row['best_metric'] = summary['best_val_rmse']
    if 'best_val_mAP' in summary:
        row['best_val_mAP'] = summary['best_val_mAP']
        row['best_metric'] = summary['best_val_mAP']

    final = summary.get('final_metrics') or {}
    for key in DEPTH_METRIC_KEYS:
        short = key[len('val_'):]
        if short in final:
            row[key] = final[short]
    for key in DET_METRIC_KEYS:
        if key in final:
            row[key] = final[key]

    row['spike_rate'] = summary.get('overall_spike_rate')
    energy = summary.get('energy') or {}
    synaptic = energy.get('synaptic_ops') or {}
    cost = energy.get('energy') or {}
    latency = energy.get('latency_ms') or {}
    row['spikes_per_frame'] = synaptic.get('spikes_per_frame')
    row['sops_per_frame'] = synaptic.get('sops_per_frame')
    row['ann_macs_per_frame'] = energy.get('ann_macs_per_frame')
    row['mac_equivalent_ratio'] = energy.get('mac_equivalent_ratio')
    row['ann_joules_per_frame'] = cost.get('ann_joules_per_frame')
    row['snn_joules_per_frame'] = cost.get('snn_joules_per_frame')
    row['energy_ratio_ann_over_snn'] = cost.get('energy_ratio_ann_over_snn')
    row['latency_snn_T1_ms'] = latency.get('snn_T1')

    return {k: v for k, v in row.items() if v is not None}


def _clean_number(value):
    """JSON round-trip guard: reject booleans/strings masquerading as numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def collect_rows(results_root):
    """Every summary.json under the root, at any nesting depth.

    Nested directories act as namespaces (e.g. ``results/_smoke/<tag>/``);
    the tag is the path below the root joined with '/' so rows stay unique.
    Legacy flat summaries (<tag>_summary.json directly in the root) are
    still picked up for backwards compatibility.
    """
    rows = []
    flat_tags = set()

    for dirpath, _dirnames, filenames in os.walk(results_root):
        for filename in filenames:
            if filename != 'summary.json':
                continue
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, results_root)
            parts = rel.split(os.sep)[:-1]
            tag = '/'.join(parts) if len(parts) > 1 else parts[0]
            if len(parts) == 1:
                flat_tags.add(tag)
            rows.extend(_load(full, tag))

    # Legacy flat layout: results/<tag>_summary.json
    pattern_flat = os.path.join(results_root, '*_summary.json')
    for path in sorted(glob.glob(pattern_flat)):
        tag = os.path.basename(path)[:-(len('_summary.json'))]
        if tag in flat_tags:
            continue
        rows.extend(_load(path, tag))

    for row in rows:
        for key in ('best_val_rmse', 'best_val_mAP', 'best_metric'):
            row[key] = _clean_number(row.get(key))
    return rows


def _load(path, tag):
    try:
        with open(path, encoding='utf-8') as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[aggregate] warning: skipping unreadable {path} ({exc})')
        return []
    rows = [flatten_summary(summary)]
    for row in rows:
        if not row.get('tag'):
            row['tag'] = tag
            row['group'] = SEED_SUFFIX.sub('', tag)
    return rows


def write_csv(rows, path):
    if not rows:
        print(f'[aggregate] no runs found; nothing written to {path}')
        return
    ordered = ['track'] + CONFIG_KEYS + ['tag', 'group', 'best_epoch',
                                         'best_val_rmse', 'best_val_mAP',
                                         'best_metric'] \
        + DEPTH_METRIC_KEYS + DET_METRIC_KEYS + ENERGY_KEYS
    fieldnames = [k for k in ordered if any(k in r for r in rows)]
    extras = sorted({k for r in rows for k in r} - set(fieldnames))
    fieldnames += extras

    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print(f'[aggregate] {len(rows)} run(s) -> {path}')


def _fmt_cell(values, higher_better=False):
    """mean +/- std cell from a list of numbers; best-of-group marker later."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return ''
    avg = mean(values)
    spread = f' +/- {stdev(values):.4f}' if len(values) > 1 else ''
    return f'{avg:.4f}{spread}'


def write_markdown(rows, path):
    lines = ['# ViSNN experiment results', '',
             f'_Generated {datetime.now().isoformat(timespec="seconds")} '
             f'from {len(rows)} run(s)._',
             '',
             'Values are mean +/- std over seeds; single-seed cells show the '
             'bare mean. Empty cells were not reported by those runs.', '']

    tracks = []
    for row in rows:
        track = row.get('track') or ('ssd' if 'mAP' in row else 'depth')
        if track not in tracks:
            tracks.append(track)

    for track in sorted(tracks):
        title = ('Track A - depth estimation' if track == 'depth'
                 else 'Track B - SSD detection')
        lines += [f'## {title}', '']

        track_rows = [r for r in rows
                      if (r.get('track') or track) == track]
        groups = {}
        for row in track_rows:
            groups.setdefault(row['group'], []).append(row)

        metric_keys = ([k for k in DEPTH_METRIC_KEYS if track == 'depth'] +
                       [k for k in DET_METRIC_KEYS if track == 'ssd'])
        columns = (metric_keys + ['best_metric', 'spike_rate', 'sops_per_frame',
                                  'snn_joules_per_frame'])

        header = ['group', 'runs'] + columns
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('|' + '---|' * len(header))

        def sort_key(group):
            sample = groups[group][0]
            return sample.get('best_metric') if sample.get('best_metric') \
                is not None else float('-inf')

        reverse = track == 'ssd'
        for group in sorted(groups, key=sort_key, reverse=reverse):
            group_rows = groups[group]
            cells = [f'`{group}`', str(len(group_rows))]
            for col in columns:
                cells.append(_fmt_cell([r.get(col) for r in group_rows]))
            lines.append('| ' + ' | '.join(cells) + ' |')

        seeds_note = ', '.join(
            str(r.get('seed')) for r in track_rows if r.get('seed'))
        if seeds_note:
            lines += ['', f'Seeds present: {seeds_note}.']
        lines.append('')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(f'[aggregate] report -> {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default=None)
    parser.add_argument('--out-md', default=None,
                        help='markdown output path (default: '
                             '<results-root>/RESULTS.md)')
    parser.add_argument('--out-csv', default=None,
                        help='csv output path (default: '
                             '<results-root>/all_runs.csv)')
    parser.add_argument('--stdout', action='store_true',
                        help='also print the markdown report')
    args = parser.parse_args()

    import config
    root = args.results_root or config.RESULTS_DIR
    if not os.path.isdir(root):
        print(f'[aggregate] results root {root!r} does not exist yet')
        return 1

    rows = collect_rows(root)
    csv_path = args.out_csv or os.path.join(root, 'all_runs.csv')
    md_path = args.out_md or os.path.join(root, 'RESULTS.md')
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    if args.stdout:
        with open(md_path, encoding='utf-8') as handle:
            print(handle.read())
    return 0


if __name__ == '__main__':
    sys.exit(main())
