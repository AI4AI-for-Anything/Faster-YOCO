"""Fused RMSNorm forward/backward in Triton (switch YOCO_TRITON_NORM=1), the upstream expression
    n = bf16(x.float() * rsqrt(mean(x.float()^2) + eps));  out = n * w      (w optional)
Forward: one kernel per row block (fp32 statistics, the same bf16 roundings as upstream: n is rounded to the input
dtype before the weight multiply, the product is rounded once). Backward: dx in one kernel (fp32 math on dn = dy*w
rounded to bf16, as upstream's autograd sees it); the weight gradient sum(dy * n) is accumulated in fp32 per program
and summed over programs by a second reduction (no atomics: deterministic).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(X, W, Y, RSTD, N, eps, stride, HAS_W: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    n = (x * r).to(Y.dtype.element_ty)
    if HAS_W:
        w = tl.load(W + cols, mask=mask, other=0.0)
        y = (n.to(tl.float32) * w.to(tl.float32)).to(Y.dtype.element_ty)
    else:
        y = n
    tl.store(Y + row * stride + cols, y, mask=mask)
    tl.store(RSTD + row, r)


@triton.jit
def _bwd(DY, X, W, RSTD, DX, DWP, M, N, stride, rows_per_prog, HAS_W: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    if HAS_W:
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dw = tl.zeros([BLOCK], dtype=tl.float32)
    start = pid * rows_per_prog
    for i in range(0, rows_per_prog):
        row = start + i
        if row < M:
            x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(RSTD + row)
            nf = x * r
            if HAS_W:
                nb = nf.to(DX.dtype.element_ty).to(tl.float32)
                dw += dy * nb
                dn = (dy * w).to(DX.dtype.element_ty).to(tl.float32)
            else:
                dn = dy
            c = tl.sum(dn * nf, axis=0) / N
            dx = r * (dn - nf * c)
            tl.store(DX + row * stride + cols, dx.to(DX.dtype.element_ty), mask=mask)
    if HAS_W:
        tl.store(DWP + pid * N + cols, dw, mask=mask)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        M, N = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        BLOCK = triton.next_power_of_2(N)
        has_w = weight is not None
        _fwd[(M,)](x2, weight if has_w else x2, y, rstd, N, eps, x2.stride(0), HAS_W=has_w, BLOCK=BLOCK,
                   num_warps=8 if BLOCK >= 2048 else 4)
        ctx.save_for_backward(x2, weight, rstd) if has_w else ctx.save_for_backward(x2, rstd)
        ctx.has_w, ctx.shape, ctx.N = has_w, shape, N
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        if ctx.has_w:
            x2, weight, rstd = ctx.saved_tensors
        else:
            (x2, rstd), weight = ctx.saved_tensors, None
        M, N = x2.shape
        dy2 = dy.reshape(M, N)
        if dy2.stride(-1) != 1 or dy2.stride(0) != x2.stride(0):
            dy2 = dy2.contiguous()
        dx = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        progs = min(M, 4 * torch.cuda.get_device_properties(x2.device).multi_processor_count)
        rows = (M + progs - 1) // progs
        progs = (M + rows - 1) // rows
        dwp = torch.empty(progs, N, dtype=torch.float32, device=x2.device) if ctx.has_w else dx
        _bwd[(progs,)](dy2, x2, weight if ctx.has_w else x2, rstd, dx, dwp, M, N, x2.stride(0), rows,
                       HAS_W=ctx.has_w, BLOCK=BLOCK, num_warps=8 if BLOCK >= 2048 else 4)
        dw = dwp.sum(0).to(weight.dtype) if ctx.has_w else None
        return dx.view(ctx.shape), dw, None


def rms_norm(x, weight, eps):
    return _RMSNorm.apply(x, weight, eps)
