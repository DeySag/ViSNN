"""Calibration: activation profiling and scaling-factor search."""

from calibration.lambda_search import measure_spike_rate, search_lambda
from calibration.profile import (
    ActivationProfiler,
    collect_profiles,
    compute_channel_thresholds,
    load_thresholds,
    save_thresholds,
    summarize_thresholds,
)

__all__ = [
    'ActivationProfiler',
    'collect_profiles',
    'compute_channel_thresholds',
    'load_thresholds',
    'measure_spike_rate',
    'save_thresholds',
    'search_lambda',
    'summarize_thresholds',
]
