import torch
import torch.nn.functional as F

from models.snn import set_lambda


def compute_feature_mse(snn_features, continuous_features):
    """Feature-level fidelity proxy between SNN and continuous encoders."""
    return F.mse_loss(snn_features, continuous_features).item()


@torch.no_grad()
def search_lambda(snn_encoder, golden_encoder, loader, device,
                  grid=(0.1, 0.25, 0.5, 0.75, 1.0), num_batches=5):
    """Grid-search the global SFN scaling factor lambda using a feature-MSE
    proxy on the calibration/validation set. Returns (best_lambda, results)."""
    continuous = []
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        continuous.append(golden_encoder(images.to(device)).cpu())
    continuous = torch.cat(continuous, dim=0)
    print(f'Cached continuous features: {tuple(continuous.shape)}')

    results = {}
    best_lambda, best_mse = None, float('inf')
    for lam in grid:
        set_lambda(snn_encoder, lam)
        per_bit = []
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            per_bit.append(snn_encoder(images.to(device)).cpu())
        snn_features = torch.cat(per_bit, dim=0)
        mse = compute_feature_mse(snn_features, continuous)
        results[lam] = mse
        print(f'  lambda={lam:.3f} -> feature MSE={mse:.6f}')
        if mse < best_mse:
            best_mse, best_lambda = mse, lam

    set_lambda(snn_encoder, best_lambda)
    return best_lambda, results