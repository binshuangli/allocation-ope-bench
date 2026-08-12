"""Global seed control — call once at the start of every script/experiment."""

import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed Python random, NumPy, and (if available) PyTorch and TF."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
