"""Theoretical MAC -> AC energy estimator for the spiking pipeline.

SNN inference replaces multiply-accumulate (MAC) with accumulate-only (AC)
operations plus a threshold comparison at each spiking layer. Energy is only
a hardware property; these figures estimate the relative platform saving and
are reported as a ratio, not absolute deployment power.

Representative pJ/op at 45nm CMOS (Horowitz 2014), used widely in SNN
literature (e.g. Spike-YOLO, SFormer):
    E_MAC     = 4.6 pJ  (8-bit multiply-accumulate)
    E_AC      = 0.9 pJ  (8-bit accumulate-only)
    E_COMPARE = 0.05 pJ (threshold compare per spiking output)
"""

import torch

from models.snn import SFNNeuron

E_MAC = 4.6
E_AC = 0.9
E_COMPARE = 0.05


@torch.no_grad()
def estimate_energy(model, input_tensor):
    """Estimate ANN vs SNN energy for one forward pass.

    Returns (ann_energy_pJ, snn_energy_pJ, ac_fraction, firing_rate).
    """
    macs = []
    rates = []
    handles = []

    def conv_hook(module, _, out):
        _, c_out, h, w = out.shape
        macs.append(c_out * h * w * module.kernel_size[0] ** 2 * module.in_channels)

    def sfn_hook(_, __, out):
        rates.append((out.abs() > 0).float().mean().item())

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, SFNNeuron):
            handles.append(module.register_forward_hook(sfn_hook))

    model(input_tensor)

    for handle in handles:
        handle.remove()

    total_macs = sum(macs) if macs else 0
    snn_acs = 0.0
    for i, m in enumerate(macs):
        rate = rates[i] if i < len(rates) else 1.0
        snn_acs += m * rate

    firing_rate = sum(rates) / len(rates) if rates else 1.0
    ann_energy = total_macs * E_MAC
    snn_energy = snn_acs * E_AC + total_macs * E_COMPARE
    return ann_energy, snn_energy, snn_acs / max(total_macs, 1), firing_rate