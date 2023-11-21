from typing import Any, Optional

import torch as th

DEFAULT_DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
DEFAULT_DTYPE = th.float32


def get_tensor(
    x: Any,
    dtype: Optional[th.dtype] = None,
    device: Optional[th.device] = None,
    requires_grad: bool = False,
) -> th.Tensor:
    dtype = DEFAULT_DTYPE
    device = DEFAULT_DEVICE if device is None else device

    return th.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
