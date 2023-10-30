import random

import numpy as np
import torch


def set_deterministic_torch(seed: int = 0) -> None:
    """
    Set seeds and configurations to enable deterministic behavior in PyTorch.

    Parameters
    ----------
    seed : int
        Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
