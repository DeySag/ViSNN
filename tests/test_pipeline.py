"""End-to-end verification of both tracks.

Runs entirely on synthetic data with randomly-initialised backbones, so it
needs no dataset download and no network access:

    python tests/test_pipeline.py

Each check either prints `ok` or raises. The point is to catch the failure
modes that are silent in this kind of pipeline -- misaligned depth targets,
prior/head count mismatches, a backbone that never actually got converted, an
"frozen" encoder that is still learning -- not just that the code imports.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils
from calibration.profile import collect_profiles
from data.loaders import build_depth_loaders, build_detection_loaders
from data.synthetic import SyntheticDepthDataset, SyntheticDetectionDataset
from losses.depth_loss import compute_depth_loss, masked_rmse, valid_mask
from losses.multibox_loss import MultiBoxLoss
from models.backbone import MobileNetV2Backbone
from models.box_utils import decode, encode, jaccard, match_priors, nms
from models.snn import (
    StrictT1SFN,
    assert_frozen,
    convert_to_snn,
    count_spiking_layers,
    freeze_module,
)
from models.ssd import PriorBox, SpikingSSD, sanity_check_ssd
from pipeline import build_depth_pipeline, build_ssd_pipeline
from validation.metrics_depth import DepthMetrics, evaluate_depth
from validation.metrics_detection import DetectionEvaluator, evaluate_detection
from validation.visualize import (
    plot_curves,
    visualize_depth_batch,
    visualize_detection_batch,
)

DEVICE = torch.device('cpu')
PASSED, FAILED = [], []


def check(name):
    """Decorator that runs a check immediately and records the outcome."""
    def wrapper(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- report, don't abort the suite
            FAILED.append((name, exc))
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
            import traceback
            traceback.print_exc()
        else:
            PASSED.append(name)
            print(f'  ok    {name}')
        return fn
    return wrapper


# ===========================================================================
print('\n[1] Data pipeline')
# ===========================================================================

@check('synthetic depth sample has the contracted shape/dtype')
def _():
    dataset = SyntheticDepthDataset(length=4)
    image, depth = dataset[0]
    assert image.shape == (3, 224, 224), image.shape
    assert depth.shape == (1, 224, 224), depth.shape
    assert image.dtype == torch.float32 and depth.dtype == torch.float32
    assert depth.min() >= 0.0 and depth.max() <= config.DEPTH_MAX


@check('depth targets are sparse and stay within the KITTI range')
def _():
    dataset = SyntheticDepthDataset(length=4)
    _, depth = dataset[0]
    fraction = valid_mask(depth).float().mean().item()
    assert 0.02 < fraction < 0.6, f'valid fraction {fraction} looks wrong'


@check('depth dataset is deterministic for a fixed index')
def _():
    a = SyntheticDepthDataset(length=4)[2]
    b = SyntheticDepthDataset(length=4)[2]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@check('CenterCrop keeps RGB and depth geometrically aligned')
def _():
    # Render a frame whose depth is an exact function of the pixel column, then
    # confirm the cropped depth matches the analytically expected crop window.
    from PIL import Image
    from data.transforms import DepthJointTransform

    width, height = 1242, 375
    columns = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    depth = columns / width * 50.0 + 1.0
    image = Image.new('RGB', (width, height))

    transform = DepthJointTransform(size=224, mode='crop', augment=False)
    _, depth_t = transform(image, depth)

    left = (width - 224) // 2
    expected_left = (left / width * 50.0 + 1.0)
    expected_right = ((left + 223) / width * 50.0 + 1.0)
    assert abs(depth_t[0, 0, 0].item() - expected_left) < 1e-3, depth_t[0, 0, 0]
    assert abs(depth_t[0, 0, -1].item() - expected_right) < 1e-3


@check('detection boxes stay normalized and non-degenerate')
def _():
    dataset = SyntheticDetectionDataset(length=4)
    image, boxes, labels = dataset[0]
    assert image.shape == (3, 224, 224)
    assert boxes.shape[1] == 4 and boxes.shape[0] == labels.shape[0]
    assert boxes.min() >= 0.0 and boxes.max() <= 1.0
    assert ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()
    assert labels.min() >= 1


@check('loaders batch correctly (stacked images, per-image box lists)')
def _():
    train_loader, val_loader = build_depth_loaders(
        synthetic=True, batch_size=4, num_workers=0, max_train=8, max_val=4)
    images, depths = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224) and depths.shape == (4, 1, 224, 224)
    assert len(val_loader.dataset) == 4

    det_loader, _ = build_detection_loaders(
        synthetic=True, batch_size=4, num_workers=0, max_train=8, max_val=4)
    images, boxes, labels = next(iter(det_loader))
    assert images.shape == (4, 3, 224, 224)
    assert len(boxes) == 4 and len(labels) == 4


# ===========================================================================
print('\n[2] SNN neuron and conversion surgery')
# ===========================================================================

@check('StrictT1SFN fires exactly at the per-channel threshold')
def _():
    neuron = StrictT1SFN([1.0, 2.0], lambda_=1.0, fire_fn='binary')
    x = torch.tensor([[[[0.5]], [[2.5]]]])          # below theta, above theta
    out = neuron(x)
    assert out[0, 0, 0, 0].item() == 0.0
    assert abs(out[0, 1, 0, 0].item() - 2.0) < 1e-6

    x_edge = torch.tensor([[[[1.0]], [[2.0]]]])     # exactly at threshold
    out_edge = neuron(x_edge)
    assert abs(out_edge[0, 0, 0, 0].item() - 1.0) < 1e-6


@check('lambda scales the effective threshold')
def _():
    neuron = StrictT1SFN([2.0], lambda_=0.25)
    x = torch.tensor([[[[0.6]]]])                   # below 2.0, above 0.5
    assert abs(neuron(x)[0, 0, 0, 0].item() - 0.5) < 1e-6


@check('output is strictly binary-valued per channel (T=1)')
def _():
    neuron = StrictT1SFN([1.5] * 3)
    out = neuron(torch.rand(2, 3, 8, 8) * 4)
    unique = torch.unique(out)
    assert set(round(v, 4) for v in unique.tolist()) <= {0.0, 1.5}


@check('MTN mode produces graded, bounded levels')
def _():
    neuron = StrictT1SFN([1.0], fire_fn='mtn', n_levels=4)
    out = neuron(torch.tensor([[[[0.5, 1.2, 3.7, 99.0]]]]))
    assert out.flatten().tolist() == [0.0, 1.0, 3.0, 4.0]


@check('multi-timestep membrane resets by subtraction')
def _():
    neuron = StrictT1SFN([1.0], timesteps=4)
    neuron.reset_membrane()
    x = torch.tensor([[[[0.6]]]])
    outs = [neuron(x)[0, 0, 0, 0].item() for _ in range(4)]
    # 0.6, 1.2->fire (residual .2), 0.8, 1.4->fire  => rate 0.5
    assert outs == [0.0, 1.0, 0.0, 1.0], outs


@check('conversion replaces ReLU6 in MobileNetV2 (not just ReLU)')
def _():
    backbone = MobileNetV2Backbone(pretrained=False)
    relu6_count = sum(1 for m in backbone.modules() if isinstance(m, nn.ReLU6))
    assert relu6_count > 0, 'test premise broken: no ReLU6 found'

    loader, _ = build_depth_loaders(synthetic=True, batch_size=2,
                                    num_workers=0, max_train=4, max_val=2)
    profiler = collect_profiles(backbone, loader, DEVICE, num_batches=2,
                                verbose=False)
    thresholds = profiler.thresholds(percentile=99.0)
    replaced = convert_to_snn(backbone, thresholds, device=DEVICE)

    assert replaced == relu6_count, f'{replaced} replaced vs {relu6_count} found'
    assert count_spiking_layers(backbone) == relu6_count
    assert sum(1 for m in backbone.modules()
               if isinstance(m, nn.ReLU6)) == 0


@check('calibrated thresholds are positive and per-channel')
def _():
    backbone = MobileNetV2Backbone(pretrained=False)
    loader, _ = build_depth_loaders(synthetic=True, batch_size=2,
                                    num_workers=0, max_train=4, max_val=2)
    profiler = collect_profiles(backbone, loader, DEVICE, num_batches=2,
                                verbose=False)
    thresholds = profiler.thresholds()
    assert thresholds, 'no thresholds captured'
    for name, theta in thresholds.items():
        assert theta.ndim == 1 and len(theta) > 1, name
        assert (theta > 0).all(), f'{name} has a non-positive threshold'


@check('freeze_module blocks gradients and pins BatchNorm')
def _():
    backbone = MobileNetV2Backbone(pretrained=False)
    freeze_module(backbone)
    assert_frozen(backbone)
    assert not backbone.training


# ===========================================================================
print('\n[3] Losses')
# ===========================================================================

@check('masked RMSE ignores invalid (zero) ground-truth pixels')
def _():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, 0] = 10.0                      # one valid pixel
    predicted = torch.full((1, 1, 4, 4), 12.0)
    # Error is 2.0 on the only valid pixel; the 15 zeros must not participate.
    assert abs(masked_rmse(predicted, target).item() - 2.0) < 1e-3


@check('depth loss is finite and positive when the model over-predicts')
def _():
    # The `(pred - target) * 2` formulation goes NaN here; `** 2` does not.
    target = torch.ones(2, 1, 16, 16) * 5.0
    predicted = torch.ones(2, 1, 16, 16) * 9.0
    loss = compute_depth_loss(predicted, target)
    assert torch.isfinite(loss) and loss.item() > 0, loss


@check('depth loss backpropagates without NaN at a perfect fit')
def _():
    target = torch.ones(1, 1, 8, 8) * 3.0
    predicted = (torch.ones(1, 1, 8, 8) * 3.0).requires_grad_(True)
    compute_depth_loss(predicted, target).backward()
    assert torch.isfinite(predicted.grad).all(), 'sqrt(0) produced a NaN grad'


@check('depth loss survives a batch with no valid pixels')
def _():
    target = torch.zeros(1, 1, 8, 8)
    predicted = torch.rand(1, 1, 8, 8, requires_grad=True)
    loss = compute_depth_loss(predicted, target)
    loss.backward()
    assert torch.isfinite(loss)


@check('MultiBox loss is finite and rewards a correct prediction')
def _():
    priorbox = PriorBox((14, 7, 4, 2, 1))
    priors = priorbox.priors
    criterion = MultiBoxLoss(priors, num_classes=4)

    gt_boxes = [torch.tensor([[0.2, 0.2, 0.6, 0.6]])]
    gt_labels = [torch.tensor([1])]

    loc_t, label_t = criterion.build_targets(gt_boxes, gt_labels, DEVICE)
    assert (label_t > 0).sum() > 0, 'matching produced no positives'

    # A "perfect" head: exact regression targets, confident correct classes.
    perfect_loc = loc_t
    perfect_scores = torch.zeros(1, priors.shape[0], 4)
    perfect_scores.scatter_(2, label_t.unsqueeze(-1), 20.0)

    random_loc = torch.randn(1, priors.shape[0], 4)
    random_scores = torch.randn(1, priors.shape[0], 4)

    good = criterion(perfect_loc, perfect_scores, gt_boxes, gt_labels)
    bad = criterion(random_loc, random_scores, gt_boxes, gt_labels)
    assert torch.isfinite(good) and torch.isfinite(bad)
    assert good.item() < bad.item(), f'good {good.item()} !< bad {bad.item()}'


@check('hard-negative mining bounds the number of scored negatives')
def _():
    priorbox = PriorBox((14, 7, 4, 2, 1))
    criterion = MultiBoxLoss(priorbox.priors, num_classes=4, neg_pos_ratio=3)
    gt_boxes = [torch.tensor([[0.3, 0.3, 0.5, 0.5]])]
    gt_labels = [torch.tensor([2])]
    _loc_t, label_t = criterion.build_targets(gt_boxes, gt_labels, DEVICE)
    num_pos = int((label_t > 0).sum())
    assert 0 < num_pos < priorbox.priors.shape[0] * 0.5


# ===========================================================================
print('\n[4] Box maths')
# ===========================================================================

@check('encode/decode round-trips exactly')
def _():
    priors = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.25, 0.75, 0.4, 0.1]])
    boxes = torch.tensor([[0.42, 0.38, 0.63, 0.61], [0.1, 0.7, 0.45, 0.82]])
    recovered = decode(encode(boxes, priors), priors)
    assert torch.allclose(boxes, recovered, atol=1e-5), recovered


@check('IoU is correct on a known overlap')
def _():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    # intersection 1, union 4+4-1 = 7
    assert abs(jaccard(a, b).item() - 1 / 7) < 1e-6


@check('NMS suppresses duplicates but keeps distinct boxes')
def _():
    boxes = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [0.05, 0.05, 1.0, 1.0],     # near-duplicate of the first
        [5.0, 5.0, 6.0, 6.0],       # disjoint
    ])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2], keep.tolist()


@check('every ground-truth box claims at least one prior')
def _():
    priors = PriorBox((14, 7, 4, 2, 1)).priors
    # A tiny box that no prior overlaps well -- pass 2 must still match it.
    gt = torch.tensor([[0.50, 0.50, 0.52, 0.52]])
    labels = torch.tensor([3])
    _loc, matched = match_priors(gt, labels, priors)
    assert (matched == 3).sum() >= 1, 'small object left unmatched'


@check('prior count matches the head output length')
def _():
    model = SpikingSSD(num_classes=4, pretrained=False)
    info = sanity_check_ssd(model, device='cpu')
    assert info['num_priors'] == 1194, info
    assert info['loc'][1] == 1194 and info['scores'][1] == 1194


# ===========================================================================
print('\n[5] Metrics')
# ===========================================================================

@check('DepthMetrics reproduces a hand-computed RMSE')
def _():
    target = torch.zeros(1, 1, 2, 2)
    target[0, 0, 0, 0] = 4.0
    target[0, 0, 1, 1] = 10.0
    predicted = torch.zeros(1, 1, 2, 2)
    predicted[0, 0, 0, 0] = 5.0       # error 1
    predicted[0, 0, 1, 1] = 13.0      # error 3
    metrics = DepthMetrics()
    metrics.update(predicted, target)
    expected = ((1 ** 2 + 3 ** 2) / 2) ** 0.5
    assert abs(metrics.compute()['rmse'] - expected) < 1e-6


@check('mAP is 1.0 for perfect detections')
def _():
    evaluator = DetectionEvaluator(num_classes=3)
    boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]])
    labels = torch.tensor([1, 2])
    evaluator.update(
        [{'boxes': boxes, 'scores': torch.tensor([0.9, 0.8]), 'labels': labels}],
        [boxes], [labels])
    assert abs(evaluator.compute()['mAP'] - 1.0) < 1e-6


@check('mAP is 0.0 when every prediction misses')
def _():
    evaluator = DetectionEvaluator(num_classes=3)
    gt_boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
    pred_boxes = torch.tensor([[0.7, 0.7, 0.9, 0.9]])
    evaluator.update(
        [{'boxes': pred_boxes, 'scores': torch.tensor([0.9]),
          'labels': torch.tensor([1])}],
        [gt_boxes], [torch.tensor([1])])
    assert evaluator.compute()['mAP'] == 0.0


@check('a duplicate box is matched only once (second copy becomes an FP)')
def _():
    evaluator = DetectionEvaluator(num_classes=2)
    box = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    doubled = torch.cat([box, box], dim=0)
    evaluator.update(
        [{'boxes': doubled, 'scores': torch.tensor([0.9, 0.85]),
          'labels': torch.tensor([1, 1])}],
        [box], [torch.tensor([1])])
    # The single ground-truth box can only be credited once, so the second
    # detection is scored as a false positive. AP is still 1.0 here, and that
    # is correct: the duplicate ranks *below* the true positive and therefore
    # adds no recall, so it never enters the precision-recall integral. This is
    # standard VOC/COCO behaviour, not a bug -- the penalty for a duplicate
    # only materialises when it outranks a true positive (next check).
    assert abs(evaluator.compute()['mAP'] - 1.0) < 1e-6


@check('a false positive outranking a true positive halves AP')
def _():
    evaluator = DetectionEvaluator(num_classes=2)
    gt = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    predictions = torch.tensor([
        [0.7, 0.7, 0.95, 0.95],     # confident and wrong -> FP, ranked first
        [0.1, 0.1, 0.5, 0.5],       # correct -> TP, ranked second
    ])
    evaluator.update(
        [{'boxes': predictions, 'scores': torch.tensor([0.95, 0.90]),
          'labels': torch.tensor([1, 1])}],
        [gt], [torch.tensor([1])])
    # PR curve: (r=0, p=0) then (r=1, p=0.5) -> AP = 0.5.
    assert abs(evaluator.compute()['mAP'] - 0.5) < 1e-6


@check('missed ground truth caps recall and therefore AP')
def _():
    evaluator = DetectionEvaluator(num_classes=2)
    gt = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]])
    found = torch.tensor([[0.1, 0.1, 0.4, 0.4]])     # only one of the two
    evaluator.update(
        [{'boxes': found, 'scores': torch.tensor([0.9]),
          'labels': torch.tensor([1])}],
        [gt], [torch.tensor([1, 1])])
    assert abs(evaluator.compute()['mAP'] - 0.5) < 1e-6


# ===========================================================================
print('\n[6] End-to-end: Track A')
# ===========================================================================

@check('depth pipeline builds, trains a step, and improves the loss')
def _():
    utils.set_seed(0)
    train_loader, val_loader = build_depth_loaders(
        synthetic=True, batch_size=4, num_workers=0, max_train=8, max_val=4)

    model, thresholds = build_depth_pipeline(
        train_loader, DEVICE, pretrained=False, calibration_batches=2,
        verbose=False)
    assert len(thresholds) > 0

    assert_frozen(model.encoder, 'encoder')
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n.startswith('head.') for n in trainable), trainable

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    images, depths = next(iter(train_loader))

    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features = model.encoder(images)
        predicted = model.decode(features)
        assert predicted.shape == depths.shape, predicted.shape
        loss = compute_depth_loss(predicted, depths)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], f'loss did not decrease: {losses[0]} -> {losses[-1]}'

    metrics = evaluate_depth(model, val_loader, DEVICE)
    assert np.isfinite(metrics['rmse']) and metrics['rmse'] >= 0


@check('frozen encoder weights are byte-identical after training')
def _():
    utils.set_seed(0)
    train_loader, _ = build_depth_loaders(
        synthetic=True, batch_size=4, num_workers=0, max_train=8, max_val=4)
    model, _ = build_depth_pipeline(train_loader, DEVICE, pretrained=False,
                                    calibration_batches=2, verbose=False)

    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-2)
    images, depths = next(iter(train_loader))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        compute_depth_loss(model(images), depths).backward()
        optimizer.step()

    after = model.encoder.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), f'{key} drifted while frozen'


# ===========================================================================
print('\n[7] End-to-end: Track B')
# ===========================================================================

@check('ssd pipeline builds, trains a step, and improves the loss')
def _():
    utils.set_seed(0)
    train_loader, val_loader = build_detection_loaders(
        synthetic=True, batch_size=4, num_workers=0, max_train=8, max_val=4)
    num_classes = train_loader.dataset.num_classes

    model, thresholds = build_ssd_pipeline(
        train_loader, DEVICE, num_classes=num_classes, pretrained=False,
        calibration_batches=2, verbose=False)
    assert len(thresholds) > 0
    assert_frozen(model.backbone.stage1, 'stage1')
    assert_frozen(model.backbone.stage2, 'stage2')

    criterion = MultiBoxLoss(model.priors, num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    images, boxes, labels = next(iter(train_loader))

    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loc, scores = model(images)
        loss = criterion(loc, scores, boxes, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], f'loss did not decrease: {losses[0]} -> {losses[-1]}'

    metrics = evaluate_detection(model, val_loader, DEVICE,
                                 num_classes=num_classes)
    assert 0.0 <= metrics['mAP'] <= 1.0


@check('detect() returns well-formed, in-range detections')
def _():
    utils.set_seed(0)
    model = SpikingSSD(num_classes=4, pretrained=False)
    model.eval()
    detections = model.detect(torch.randn(2, 3, 224, 224), score_threshold=0.05)
    assert len(detections) == 2
    for det in detections:
        assert det['boxes'].shape[0] == det['scores'].shape[0]
        assert det['boxes'].shape[0] == det['labels'].shape[0]
        if det['boxes'].numel():
            assert det['boxes'].min() >= 0.0 and det['boxes'].max() <= 1.0
            assert (det['labels'] >= 1).all(), 'background emitted as detection'


# ===========================================================================
print('\n[8] Visualisation')
# ===========================================================================

@check('visualisation writes readable figures for both tracks')
def _():
    out_dir = os.path.join(config.RESULTS_DIR, 'selftest')
    os.makedirs(out_dir, exist_ok=True)

    images = torch.randn(2, 3, 224, 224)
    targets = torch.rand(2, 1, 224, 224) * 30
    targets[targets < 15] = 0.0                 # simulate LiDAR sparsity
    predictions = torch.rand(2, 1, 224, 224) * 30
    path = visualize_depth_batch(images, targets, predictions,
                                 os.path.join(out_dir, 'depth.png'),
                                 num_samples=2)
    assert path and os.path.getsize(path) > 1000

    detections = [{'boxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
                   'scores': torch.tensor([0.87]),
                   'labels': torch.tensor([1])} for _ in range(2)]
    gt_boxes = [torch.tensor([[0.12, 0.12, 0.48, 0.52]]) for _ in range(2)]
    gt_labels = [torch.tensor([1]) for _ in range(2)]
    path = visualize_detection_batch(images, detections, gt_boxes, gt_labels,
                                     os.path.join(out_dir, 'detect.png'),
                                     num_samples=2,
                                     class_names={1: 'box'})
    assert path and os.path.getsize(path) > 1000

    path = plot_curves({'loss': [1.0, 0.7, 0.5], 'mAP': [0.1, 0.3, 0.5]},
                       os.path.join(out_dir, 'curves.png'))
    assert path and os.path.getsize(path) > 1000


# ===========================================================================
print('\n[9] Real dataset loaders (miniature on-disk fixtures)')
# ===========================================================================

@check('KITTIDepthDataset indexes a real-layout tree and decodes 16-bit depth')
def _():
    import shutil
    import tempfile
    from PIL import Image
    from data.kitti import KITTIDepthDataset, build_index

    root = tempfile.mkdtemp(prefix='kitti_fixture_')
    try:
        # Two drives so the drive-disjoint train/val split has something to do.
        for drive in ('2011_09_26_drive_0001_sync', '2011_09_26_drive_0002_sync'):
            rgb_dir = os.path.join(root, drive, 'image_02', 'data')
            depth_dir = os.path.join(root, drive, 'proj_depth', 'groundtruth',
                                     'image_02')
            os.makedirs(rgb_dir)
            os.makedirs(depth_dir)
            for frame in range(3):
                name = f'{frame:010d}.png'
                Image.new('RGB', (1242, 375),
                          (frame * 40, 100, 150)).save(
                              os.path.join(rgb_dir, name))
                # 16-bit depth: 12.5 m encoded as 12.5 * 256 = 3200.
                raw = np.full((375, 1242), 3200, dtype=np.uint16)
                Image.fromarray(raw, mode='I;16').save(
                    os.path.join(depth_dir, name))

        pairs = build_index(root, cameras=('image_02',))
        assert len(pairs) == 6, f'indexed {len(pairs)} pairs, expected 6'

        dataset = KITTIDepthDataset(root_dir=root, split='all',
                                    cameras=('image_02',), cache_index=False)
        assert len(dataset) == 6
        image, depth = dataset[0]
        assert image.shape == (3, 224, 224) and depth.shape == (1, 224, 224)
        # 3200 / 256 == 12.5 metres, and the CenterCrop must not disturb it.
        assert abs(depth.max().item() - 12.5) < 1e-4, depth.max().item()
        assert valid_mask(depth).all(), 'dense fixture should be fully valid'

        train = KITTIDepthDataset(root_dir=root, split='train',
                                  cameras=('image_02',), cache_index=False)
        val = KITTIDepthDataset(root_dir=root, split='val',
                                cameras=('image_02',), cache_index=False)
        assert len(train) + len(val) == 6
        assert len(train) > 0 and len(val) > 0
        # No frame may appear in both splits.
        assert not (set(train.image_paths) & set(val.image_paths))
    finally:
        shutil.rmtree(root, ignore_errors=True)


@check('KITTI loader raises a clear error on a missing root')
def _():
    from data.kitti import KITTIDepthDataset
    try:
        KITTIDepthDataset(root_dir=os.path.join('nonexistent', 'kitti'))
    except FileNotFoundError as exc:
        assert 'synthetic' in str(exc), 'error should suggest the fallback'
    else:
        raise AssertionError('expected FileNotFoundError')


@check('COCODetectionDataset parses real annotation JSON and normalizes boxes')
def _():
    import json
    import shutil
    import tempfile
    from PIL import Image
    from data.coco import COCODetectionDataset

    root = tempfile.mkdtemp(prefix='coco_fixture_')
    try:
        split = 'val2017'
        os.makedirs(os.path.join(root, 'annotations'))
        os.makedirs(os.path.join(root, split))
        for image_id in (1, 2):
            Image.new('RGB', (640, 480), (30, 60, 90)).save(
                os.path.join(root, split, f'{image_id:012d}.jpg'))

        annotations = {
            'images': [
                {'id': 1, 'file_name': '000000000001.jpg',
                 'width': 640, 'height': 480},
                {'id': 2, 'file_name': '000000000002.jpg',
                 'width': 640, 'height': 480},
            ],
            # Sparse COCO ids with a gap, to exercise the dense remap.
            'categories': [{'id': 1, 'name': 'person'},
                           {'id': 3, 'name': 'car'},
                           {'id': 90, 'name': 'toothbrush'}],
            'annotations': [
                # [x, y, w, h] absolute -> expect [0.1, 0.25, 0.6, 0.75]
                {'id': 1, 'image_id': 1, 'category_id': 3,
                 'bbox': [64, 120, 320, 240], 'iscrowd': 0},
                {'id': 2, 'image_id': 1, 'category_id': 90,
                 'bbox': [0, 0, 64, 48], 'iscrowd': 0},
                # iscrowd must be dropped.
                {'id': 3, 'image_id': 2, 'category_id': 1,
                 'bbox': [10, 10, 100, 100], 'iscrowd': 1},
                {'id': 4, 'image_id': 2, 'category_id': 1,
                 'bbox': [320, 240, 320, 240], 'iscrowd': 0},
            ],
        }
        with open(os.path.join(root, 'annotations', f'instances_{split}.json'),
                  'w', encoding='utf-8') as handle:
            json.dump(annotations, handle)

        dataset = COCODetectionDataset(root_dir=root, split=split)
        assert len(dataset) == 2, len(dataset)
        assert dataset.num_classes == 4, dataset.num_classes
        # Sparse {1, 3, 90} -> dense {1, 2, 3}, background reserved at 0.
        assert dataset.coco_id_to_dense == {1: 1, 3: 2, 90: 3}
        assert dataset.dense_to_name[0] == '__background__'

        image, boxes, labels = dataset[0]
        assert image.shape == (3, 224, 224)
        assert boxes.shape == (2, 4) and labels.tolist() == [2, 3]
        expected = torch.tensor([0.1, 0.25, 0.6, 0.75])
        assert torch.allclose(boxes[0], expected, atol=1e-6), boxes[0]

        # The crowd annotation must not have produced a box.
        _, boxes2, labels2 = dataset[1]
        assert boxes2.shape[0] == 1 and labels2.tolist() == [1]
    finally:
        shutil.rmtree(root, ignore_errors=True)


@check('COCO loader raises a clear error on missing annotations')
def _():
    from data.coco import COCODetectionDataset
    try:
        COCODetectionDataset(root_dir=os.path.join('nonexistent', 'coco'))
    except FileNotFoundError as exc:
        assert 'synthetic' in str(exc), 'error should suggest the fallback'
    else:
        raise AssertionError('expected FileNotFoundError')


# ===========================================================================
print(f'\n{"=" * 62}')
print(f'{len(PASSED)} passed, {len(FAILED)} failed')
if FAILED:
    for name, exc in FAILED:
        print(f'  FAILED: {name} -> {exc}')
    sys.exit(1)
print('All checks passed.')
