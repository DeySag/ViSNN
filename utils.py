"""Small shared helpers: seeding, device selection, checkpoint I/O, timing."""

import os
import random
import time

import numpy as np
import torch

import config


def set_seed(seed=config.SEED):
    """Seed every RNG the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer=None):
    """Return the best available device, honouring an explicit override."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(module, trainable_only=True):
    params = module.parameters()
    if trainable_only:
        params = (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def save_checkpoint(state, filename):
    """Write a checkpoint into CHECKPOINT_DIR and return its full path."""
    config.ensure_dirs()
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(state, path)
    return path


def load_checkpoint(filename, map_location='cpu'):
    path = filename
    if not os.path.isabs(path):
        candidate = os.path.join(config.CHECKPOINT_DIR, filename)
        if os.path.exists(candidate):
            path = candidate
    # torch>=2.6 defaults weights_only=True; our checkpoints hold plain tensors
    # plus python scalars, so the safe loader is fine.
    return torch.load(path, map_location=map_location, weights_only=False)


class AverageMeter:
    """Running mean of a scalar (loss, metric, ...)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0


class Timer:
    """Context manager that reports wall-clock seconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False
