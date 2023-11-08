import numpy as np
import torch as th

# DEFAULT_DEVICE = th.device("cpu")
DEFAULT_DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
DEFAULT_DTYPE = th.float32


def get_tensor(
    x: np.ndarray | list | tuple | th.Tensor,
    dtype: th.dtype | None = None,
    device: th.device | None = None,
    requires_grad: bool = False,
) -> th.Tensor:
    dtype = DEFAULT_DTYPE
    device = DEFAULT_DEVICE if device is None else device

    return th.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
