"""Quantitative energy/latency accounting for ANN vs SNN inference.

The spike *rate* reported elsewhere is a relative proxy only. This module turns
telemetry into absolute numbers:

  Synaptic ops (SOPs). Each spike from a spiking layer propagates through the
  weights of the convolution(s) that consume its output, costing
  fan-out = C_in x k_h x k_w multiply-accumulate-equivalent events. Total SOPs =
  sum over spiking layers of (spikes emitted x total fan-out of consumers).
  Consumers are attributed by execution order: every convolution is credited to
  the most recently fired spiking layer before it in the forward pass. For the
  sequential MobileNetV2/SSD graphs this is exact; at residual merges it
  attributes the following conv to the nearest branch, which is the standard
  approximation used in SNN cost accounting.

  ANN baseline MACs. Counted analytically over one dummy forward with forward
  pre-hooks on every Conv2d: C_in x C_out x k_h x k_w x H_out x W_out each.
  The converted model has identical convolutions to the original ANN, so the
  same count serves as the dense baseline.

  Energy. Configurable pJ/event constants (Horowitz-style 45nm figures) turn
  MACs and SOPs into joules per frame:

      ANN : E ~ macs x pJ_per_mac
      SNN : E ~ sops x pJ_per_sop

  The defaults are deliberately conservative for the SNN side: digital
  neuromorphic cores land anywhere between ~0.1 and ~5 pJ/SOP depending on
  memory organisation; 1.0 pJ is a defensible mid-point. Override via CLI or
  arguments when targeting a specific substrate.

  Latency. Wall-clock seconds per forward pass measured on synthetic input,
  separately for T=1..T timesteps and (when available) the unconverted model.

All counts are per frame (batch averaged), so numbers are comparable across
batch sizes and T values. `build_energy_report` produces the JSON-safe dict
embedded into every run's summary.json.
"""

import time
from collections import defaultdict

import torch
import torch.nn as nn

import config
from models.snn import (
    StrictT1SFN,
    reset_spiking_state,
    set_spike_tracking,
    spike_report,
)

# Horowitz (ISSCC 2014) style 45nm constants, picojoules per event.
DEFAULT_ANN_MAC_PJ = 4.6   # fp32 multiply + add
DEFAULT_SOP_PJ = 1.0       # event-driven synaptic update, digital core


def _sync(device):
    if device is not None and getattr(device, 'type', None) == 'cuda':
        torch.cuda.synchronize()


class _ShapeHooks:
    """Forward pre-hooks that capture conv input/output shapes once."""

    def __init__(self, root):
        self.root = root
        self.handles = []
        self.convs = []          # dicts with per-conv shape info

    def __enter__(self):
        def make_hook(store):
            def hook(module, inputs):
                x = inputs[0]
                if not torch.is_tensor(x) or x.dim() != 4:
                    return
                b, c_in, _h_in, _w_in = x.shape
                c_out = module.out_channels
                kh, kw = module.kernel_size
                # Output spatial size after stride/padding/dilation.
                h_out = ((x.shape[2] + 2 * module.padding[0]
                          - module.dilation[0] * (kh - 1) - 1)
                         // module.stride[0]) + 1
                w_out = ((x.shape[3] + 2 * module.padding[1]
                          - module.dilation[1] * (kw - 1) - 1)
                         // module.stride[1]) + 1
                store.append({
                    'c_in': int(c_in), 'c_out': int(c_out),
                    'k': int(kh * kw), 'pixels': int(h_out * w_out),
                    'groups': int(module.groups),
                })
            return hook

        for module in self.root.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(
                    module.register_forward_pre_hook(make_hook(self.convs)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False


def ann_macs_per_frame(model, input_size=config.INPUT_SIZE, device='cpu'):
    """Dense-baseline MACs for one frame: sum over convs of CI*CO*k*Hout*Wout."""
    was_training = model.training
    model.eval()
    with _ShapeHooks(model) as hooks:
        with torch.no_grad():
            model(torch.zeros(1, 3, input_size, input_size, device=device))
    model.train(was_training)

    total = 0
    per_layer = []
    for i, info in enumerate(hooks.convs):
        effective_c_in = info['c_in'] // max(info['groups'], 1)
        macs = effective_c_in * info['c_out'] * info['k'] * info['pixels']
        total += macs
        per_layer.append({'index': i, **info, 'macs': macs})
    return {'total_macs': int(total), 'layers': per_layer}


@torch.no_grad()
def measure_synaptic_ops(model, loader, device, max_batches=3, timesteps=1,
                         verbose=False):
    """Spikes and attributed SOPs over a few validation batches.

    Returns per-frame quantities plus the per-layer breakdown. Uses the same
    spike telemetry buffers as `measure_spike_rate`, so the two reports always
    agree on firing statistics.
    """
    def _forward_all(x):
        # Handles both tracks' outputs: depth encoder returns a tensor, the
        # SSD backbone a list of feature maps.
        reset_spiking_state(model)
        if timesteps <= 1:
            return model(x)
        accumulated = None
        for _ in range(timesteps):
            out = model(x)
            if torch.is_tensor(out):
                accumulated = out if accumulated is None else accumulated + out
            else:
                if accumulated is None:
                    accumulated = [o.clone() for o in out]
                else:
                    accumulated = [a + o for a, o in zip(accumulated, out)]
        return accumulated

    producer = {'name': None}
    fanouts = defaultdict(float)          # SFN name -> summed consumer fan-out
    handles = []

    def conv_pre_hook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 4 or producer['name'] is None:
            return
        kh, kw = module.kernel_size
        fanouts[producer['name']] += (
            (x.shape[1] // max(module.groups, 1)) * kh * kw)

    def sfn_hook(name):
        def hook(_module, _inputs, _output):
            producer['name'] = name
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_pre_hook(conv_pre_hook))
        elif isinstance(module, StrictT1SFN):
            handles.append(module.register_forward_hook(sfn_hook(name)))

    set_spike_tracking(model, enabled=True, reset=True)
    frames = 0
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            frames += images.shape[0]
            _forward_all(images)
    finally:
        for handle in handles:
            handle.remove()
        report = spike_report(model)
        spikes = {name: m.spike_count.item() for name, m in
                  model.named_modules() if isinstance(m, StrictT1SFN)}
        elements = {name: m.element_count.item() for name, m in
                    model.named_modules() if isinstance(m, StrictT1SFN)}
        set_spike_tracking(model, enabled=False, reset=True)

    layers = {}
    total_sops = 0
    total_spikes = 0
    for name in spikes:
        layer_spikes = float(spikes[name])
        total_spikes += layer_spikes
        fanout = float(fanouts.get(name, 0.0))
        sops = layer_spikes * fanout
        total_sops += sops
        layers[name] = {
            'spikes': layer_spikes,
            'spike_rate': (layer_spikes / elements[name]) if elements[name] else 0.0,
            'fan_out': fanout,
            'sops': sops,
        }

    result = {
        'frames': frames,
        'timesteps': timesteps,
        'total_spikes': total_spikes,
        'spikes_per_frame': total_spikes / max(frames, 1),
        'overall_spike_rate': report['overall'],
        'num_spiking_layers': len(spikes),
        'total_sops': total_sops,
        'sops_per_frame': total_sops / max(frames, 1),
        'layers': layers,
    }
    if verbose:
        print(f'[energy] {result["sops_per_frame"] / 1e6:.2f} MSOP/frame '
              f'over {frames} frame(s), overall rate '
              f'{result["overall_spike_rate"]:.4f}')
    return result


@torch.no_grad()
def measure_latency(model, input_size=config.INPUT_SIZE, device='cpu',
                    batch_size=1, timesteps=1, warmup=10, iters=50):
    """Wall-clock milliseconds per forward pass on synthetic input."""
    was_training = model.training
    model.eval()
    dummy = torch.zeros(batch_size, 3, input_size, input_size, device=device)

    def step():
        if timesteps <= 1:
            model(dummy)
            return
        reset_spiking_state(model)
        for _ in range(timesteps):
            model(dummy)

    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    elapsed = time.perf_counter() - start
    model.train(was_training)
    return elapsed / max(iters, 1) * 1000.0


def estimate_energy(macs_per_frame=None, sops_per_frame=0.0,
                    ann_mac_pj=DEFAULT_ANN_MAC_PJ, sop_pj=DEFAULT_SOP_PJ):
    """Joule estimates per frame for the dense baseline and the SNN."""
    ann_joules = (macs_per_frame or 0.0) * ann_mac_pj * 1e-12
    snn_joules = (sops_per_frame or 0.0) * sop_pj * 1e-12
    return {
        'ann_mac_pj': ann_mac_pj,
        'sop_pj': sop_pj,
        'ann_joules_per_frame': ann_joules,
        'snn_joules_per_frame': snn_joules,
        'energy_ratio_ann_over_snn': (ann_joules / snn_joules
                                      if snn_joules > 0 else float('nan')),
    }


def build_energy_report(model, loader, device, input_size=config.INPUT_SIZE,
                        max_batches=3, timesteps=1, batch_size=1,
                        ann_mac_pj=DEFAULT_ANN_MAC_PJ,
                        sop_pj=DEFAULT_SOP_PJ, latency_iters=50,
                        verbose=False):
    """One-call accounting block embedded into run summaries.

    Combines SOP telemetry, analytic ANN MACs, latency measurements and the
    configurable-energy estimate into a single JSON-safe dict.
    """
    synaptic = measure_synaptic_ops(model, loader, device,
                                    max_batches=max_batches,
                                    timesteps=timesteps, verbose=verbose)
    macs = ann_macs_per_frame(model, input_size=input_size, device=device)

    latencies_ms = {}
    for t in sorted({1, int(timesteps)}):
        latencies_ms[f'snn_T{t}'] = measure_latency(
            model, input_size=input_size, device=device,
            batch_size=batch_size, timesteps=t,
            warmup=min(10, max(3, latency_iters // 5)), iters=latency_iters)

    sops_per_frame = synaptic['sops_per_frame']
    energy = estimate_energy(macs_per_frame=float(macs['total_macs']),
                             sops_per_frame=sops_per_frame,
                             ann_mac_pj=ann_mac_pj, sop_pj=sop_pj)

    return {
        'input_size': input_size,
        'synaptic_ops': synaptic,
        'ann_macs_per_frame': float(macs['total_macs']),
        # Undefined for the continuous control (no spikes); left null there.
        'mac_equivalent_ratio': (float(macs['total_macs']) / sops_per_frame
                                 if sops_per_frame > 0 else None),
        'latency_ms': latencies_ms,
        'energy': energy,
    }


def format_energy_report(report, prefix=''):
    """Compact human-readable digest for logs."""
    e = report['energy']
    syn = report['synaptic_ops']
    lines = [
        f"{prefix}ANN   : {report['ann_macs_per_frame'] / 1e9:.3f} GMAC/frame"
        f" -> {e['ann_joules_per_frame'] * 1e6:.3f} uJ/frame",
        f"{prefix}SNN@T{syn.get('timesteps', 1)}: "
        f"{syn['sops_per_frame'] / 1e6:.3f} MSOP/frame"
        f" (rate {syn['overall_spike_rate']:.4f})"
        f" -> {e['snn_joules_per_frame'] * 1e6:.3f} uJ/frame",
        f"{prefix}MAC-equiv ratio (dense/spiking): "
        f"{report['mac_equivalent_ratio']:.2f}",
    ]
    for key, ms in report['latency_ms'].items():
        lines.append(f'{prefix}latency {key}: {ms:.2f} ms/forward')
    return '\n'.join(lines)
