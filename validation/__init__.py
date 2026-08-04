"""Validation: depth and detection metrics, visualisation utilities."""

from validation.metrics_depth import (
    DepthMetrics,
    evaluate_depth,
    evaluate_depth_rmse,
    format_metrics,
)
from validation.metrics_detection import (
    DetectionEvaluator,
    evaluate_detection,
    evaluate_detection_map,
)
from validation.visualize import (
    plot_curves,
    plot_spike_rates,
    visualize_depth_batch,
    visualize_depth_model,
    visualize_detection_batch,
    visualize_detection_model,
)

__all__ = [
    'DepthMetrics',
    'DetectionEvaluator',
    'evaluate_depth',
    'evaluate_depth_rmse',
    'evaluate_detection',
    'evaluate_detection_map',
    'format_metrics',
    'plot_curves',
    'plot_spike_rates',
    'visualize_depth_batch',
    'visualize_depth_model',
    'visualize_detection_batch',
    'visualize_detection_model',
]
