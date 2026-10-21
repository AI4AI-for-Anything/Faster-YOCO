import os

import torch
import torch.nn as nn

# switch YOCO_COMPILE_NORM=1: the same expression, compiled by torch.compile (Inductor fuses the fp32 upcast, square,
# mean, rsqrt, scale, downcast and weight multiply into one kernel forward and one or two backward)
_COMPILE = os.environ.get('YOCO_COMPILE_NORM', '0') == '1'


def _rms(x, weight, eps: float):
    output = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x)
    if weight is not None:
        output = output * weight
    return output


_rms_fused = torch.compile(_rms, dynamic=False) if _COMPILE else None
# switch YOCO_TRITON_NORM=1: hand-written fused forward/backward kernels (rms_triton.py); takes precedence
_TRITON = os.environ.get('YOCO_TRITON_NORM', '0') == '1'


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if _TRITON and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            from .rms_triton import rms_norm
            return rms_norm(x, self.weight, self.eps)
        if _rms_fused is not None and x.is_cuda:
            return _rms_fused(x, self.weight, self.eps)
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
