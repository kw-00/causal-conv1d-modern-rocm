### Note:
This repository is dedicated to ensuring causal-conv1d support for ROCm 7.2+.

For the original project, see https://github.com/Dao-AILab/causal-conv1d.

# Causal depthwise conv1d with a PyTorch interface. ROCm-compatible.

Features:
- Support fp32, fp16, bf16.
- Kernel size 2, 3, 4.

## How to use

```python
from causal_conv1d import causal_conv1d_fn
```

```python
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
```

Equivalent to:
```python
import torch.nn.functional as F

F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```