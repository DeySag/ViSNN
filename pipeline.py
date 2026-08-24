"""Pipeline orchestration: calibrate -> convert -> freeze.

Both tracks follow the same four steps, so they live here rather than being
duplicated in the two training scripts:

    1. run the continuous pretrained network over a few batches and record
       per-channel activation percentiles (spatial-masked);
    2. replace every ReLU/ReLU6 with a StrictT1SFN carrying those thresholds;
    3. freeze the converted trunk -- no gradients, no BatchNorm drift;
    4. attach the continuous task head that will actually be trained.
"""

import os

import config
from calibration.profile import (
    collect_profiles,
    save_thresholds,
    summarize_thresholds,
)
from models.backbone import build_backbone
from models.decoder import SimpleDepthDecoder, sanity_check_decoder
from models.snn import (
    assert_frozen,
    assert_trainable,
    convert_to_snn,
    count_spiking_layers,
    freeze_module,
)
from models.ssd import SpikingSSD, sanity_check_ssd


def calibrate_and_convert(forward_model, hook_root, loader, device,
                          num_batches=config.CALIBRATION_BATCHES,
                          percentile=config.TOP_P,
                          crop_margin=config.CROP_MARGIN,
                          lambda_=config.LAMBDA, fire_fn=config.FIRE_FN,
                          timesteps=config.TIMESTEPS,
                          n_levels=config.N_LEVELS,
                          threshold_path=None, verbose=True):
    """Profile activations, then perform the ReLU -> StrictT1SFN conversion.

    `hook_root` is both where activations are observed and where they are
    replaced, so the layer names always line up. Returns the threshold dict.
    """
    if verbose:
        print(f'[calibrate] profiling {num_batches} batch(es) '
              f'at the p{percentile} percentile, crop margin {crop_margin}px')

    profiler = collect_profiles(
        forward_model, loader, device, num_batches=num_batches,
        hook_root=hook_root, crop_margin=crop_margin, verbose=verbose)
    thresholds = profiler.thresholds(percentile=percentile)

    if verbose:
        print(summarize_thresholds(thresholds))

    if threshold_path:
        os.makedirs(os.path.dirname(os.path.abspath(threshold_path)),
                    exist_ok=True)
        save_thresholds(thresholds, threshold_path)
        if verbose:
            print(f'[calibrate] thresholds saved -> {threshold_path}')

    replaced = convert_to_snn(
        hook_root, thresholds, device=device, lambda_=lambda_,
        fire_fn=fire_fn, timesteps=timesteps, n_levels=n_levels)

    if replaced == 0:
        raise RuntimeError(
            'SNN conversion replaced 0 activations. The calibrated layer names '
            'do not match the modules under the conversion root.')
    if verbose:
        print(f'[convert] replaced {replaced} activation(s) with StrictT1SFN '
              f'(lambda={lambda_}, fire_fn={fire_fn}, T={timesteps})')
    return thresholds


# ---------------------------------------------------------------------------
# Track A
# ---------------------------------------------------------------------------
def build_depth_pipeline(loader, device, backbone_name='mobilenet_v2',
                         pretrained=True, output_activation='relu',
                         calibration_batches=config.CALIBRATION_BATCHES,
                         percentile=config.TOP_P,
                         crop_margin=config.CROP_MARGIN,
                         lambda_=config.LAMBDA, fire_fn=config.FIRE_FN,
                         timesteps=config.TIMESTEPS,
                         n_levels=config.N_LEVELS,
                         threshold_path=None, verbose=True,
                         convert=True):
    """Return (model, thresholds) for the depth track, ready to train.

    With ``convert=False`` the calibration/conversion steps are skipped and
    the continuous pretrained backbone is frozen as-is. This is the
    continuous control row of the headline table: identical architecture,
    data and schedule, quantization isolated as the only difference.
    """
    backbone, out_channels = build_backbone(backbone_name, pretrained=pretrained)
    backbone = backbone.to(device)

    thresholds = None
    if convert:
        thresholds = calibrate_and_convert(
            backbone, backbone, loader, device,
            num_batches=calibration_batches, percentile=percentile,
            crop_margin=crop_margin, lambda_=lambda_, fire_fn=fire_fn,
            timesteps=timesteps, n_levels=n_levels,
            threshold_path=threshold_path, verbose=verbose)

    freeze_module(backbone)
    assert_frozen(backbone, 'depth backbone')

    model = SimpleDepthDecoder(backbone, in_channels=out_channels,
                               output_activation=output_activation).to(device)
    assert_trainable(model.head, 'depth decoder')
    sanity_check_decoder(model, device=device)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        kind = 'spiking' if convert else 'continuous (control)'
        print(f'[build] depth model: {kind} encoder | '
              f'{frozen / 1e6:.2f}M frozen, {trainable / 1e6:.2f}M trainable')
    return model, thresholds


# ---------------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------------
def build_ssd_pipeline(loader, device, num_classes=config.NUM_CLASSES,
                       pretrained=True,
                       calibration_batches=config.CALIBRATION_BATCHES,
                       percentile=config.TOP_P,
                       crop_margin=config.CROP_MARGIN,
                       lambda_=config.LAMBDA, fire_fn=config.FIRE_FN,
                       timesteps=config.TIMESTEPS, n_levels=config.N_LEVELS,
                       threshold_path=None, verbose=True, convert=True):
    """Return (model, thresholds) for the detection track, ready to train.

    Only the MobileNet stages are calibrated and converted; the SSD extra
    layers and heads are freshly initialised, have no meaningful activation
    statistics, and stay continuous. ``convert=False`` keeps the MobileNet
    stages continuous as well -- the control configuration for Track B.
    """
    model = SpikingSSD(num_classes=num_classes, pretrained=pretrained).to(device)

    thresholds = None
    if convert:
        # One ModuleList, reused for both profiling and conversion, so the
        # layer names are guaranteed identical between the two passes.
        trunk = model.backbone.spiking_trunk().to(device)

        thresholds = calibrate_and_convert(
            model.backbone, trunk, loader, device,
            num_batches=calibration_batches, percentile=percentile,
            crop_margin=crop_margin, lambda_=lambda_, fire_fn=fire_fn,
            timesteps=timesteps, n_levels=n_levels,
            threshold_path=threshold_path, verbose=verbose)

    freeze_module(model.backbone.stage1)
    freeze_module(model.backbone.stage2)
    assert_frozen(model.backbone.stage1, 'ssd stage1')
    assert_frozen(model.backbone.stage2, 'ssd stage2')
    assert_trainable(model.heads, 'ssd heads')
    assert_trainable(model.backbone.extra1, 'ssd extra1')

    model.train()  # re-pins the frozen trunk to eval via the train() override
    sanity_check_ssd(model, device=device)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        kind = (f'{count_spiking_layers(model.backbone)} spiking layer(s)'
                if convert else 'continuous trunk (control)')
        print(f'[build] ssd model: {kind} | {model.priors.shape[0]} priors | '
              f'{frozen / 1e6:.2f}M frozen, {trainable / 1e6:.2f}M trainable')
    return model, thresholds
