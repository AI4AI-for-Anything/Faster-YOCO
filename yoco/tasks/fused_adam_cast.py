"""Adam step fused with the bf16 cast of the updated master (used by zero1_fast with YOCO_FUSED_ADAM=1).

apex's multi_tensor_adam (what fairseq's FusedAdamV2.step calls, once per param group with [g, p, m, v] lists) reads
and writes the fp32 master, exp_avg and exp_avg_sq; zero1_fast then reads every updated master again to cast it into
the bf16 parameter buffer. Here the optimizer instance's `multi_tensor_adam` attribute is replaced by an op with the
same signature that runs, per tensor, one Triton kernel computing the same AdamFunctor expression in fp32 (m and v
bitwise equal to apex's; IEEE division and square root, so ~1% of the new masters differ from apex's fast-math build
in the last fp32 bit; bias corrections computed on the host as apex does) and also storing the bf16 copy of the new
master where the parameter lives. FusedAdamV2.step itself (state init, step counting, the
param-group loop) is unchanged.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _adam(G, P, M, V, P16, n, lr, beta1, beta2, eps, bc1, bc2, wd, MODE: tl.constexpr, HAS16: tl.constexpr,
          BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(G + offs, mask=mask, other=0.0)
    p = tl.load(P + offs, mask=mask, other=0.0)
    m = tl.load(M + offs, mask=mask, other=0.0)
    v = tl.load(V + offs, mask=mask, other=0.0)
    if MODE == 0:
        g = g + wd * p
    # the contraction nvcc emits for apex's `beta1 * m + (1-beta1) * g` / `beta2 * v + (1-beta2) * g * g`
    # (bitwise equal m and v in runs/opus_tools/test_adam_fma.py)
    m = tl.math.fma(beta1, m, (1 - beta1) * g)
    v = tl.math.fma(beta2, v, ((1 - beta2) * g) * g)
    mh = tl.math.div_rn(m, bc1)
    vh = tl.math.div_rn(v, bc2)
    denom = tl.math.sqrt_rn(vh) + eps
    upd = tl.math.div_rn(mh, denom)
    if MODE == 1:
        upd = upd + wd * p
    p = p - lr * upd
    tl.store(P + offs, p, mask=mask)
    tl.store(M + offs, m, mask=mask)
    tl.store(V + offs, v, mask=mask)
    if HAS16:
        tl.store(P16 + offs, p.to(P16.dtype.element_ty), mask=mask)


class FusedAdamCast:
    """Callable with amp_C.multi_tensor_adam's signature; dest: data_ptr of an fp32 master -> bf16 tensor to write
    (FusedAdamV2.step passes p.data, a new tensor object each call, so masters are matched by address)."""

    def __init__(self, dest):
        self.dest = dest
        self.cast = set()                 # data_ptrs of masters whose bf16 copy was written since the last clear

    def __call__(self, chunk_size, noop_flag, tensor_lists, lr, beta1, beta2, eps, step, mode, bias_correction,
                 weight_decay):
        bc1 = 1.0 - math.pow(beta1, step) if bias_correction == 1 else 1.0
        bc2 = 1.0 - math.pow(beta2, step) if bias_correction == 1 else 1.0
        bc1, bc2 = float(torch.tensor(bc1, dtype=torch.float32)), float(torch.tensor(bc2, dtype=torch.float32))
        BLOCK = 2048
        for g, p, m, v in zip(*tensor_lists):
            if not (g.is_contiguous() and p.is_contiguous() and m.is_contiguous() and v.is_contiguous()
                    and g.dtype == p.dtype == m.dtype == v.dtype == torch.float32):
                raise RuntimeError('fused_adam_cast: expected contiguous fp32 tensors')
            d = self.dest.get(p.data_ptr())
            n = p.numel()
            _adam[(triton.cdiv(n, BLOCK),)](g, p, m, v, d if d is not None else p, n, lr, beta1, beta2, eps, bc1, bc2,
                                            weight_decay, MODE=int(mode), HAS16=d is not None, BLOCK=BLOCK,
                                            num_warps=8)
            if d is not None:
                self.cast.add(p.data_ptr())


@triton.jit
def _scale_up(G16, G32, F, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    f = tl.load(F)
    g = tl.load(G16 + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(G32 + offs, g * f, mask=mask)


def scale_up(g16, g32, factor):
    """g32 = float(g16) * factor (factor: 1-element fp32 tensor) -- the fp32 product of the exactly upcast value, as
    torch.mul(g16, factor, out=g32) computes it, in a vectorized kernel (the broadcast mul runs the generic one)."""
    n = g16.numel()
    BLOCK = 4096
    _scale_up[(triton.cdiv(n, BLOCK),)](g16, g32, factor, n, BLOCK=BLOCK, num_warps=8)
