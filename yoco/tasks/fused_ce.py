"""Fused cross-entropy for fairseq's CrossEntropyCriterion (switch YOCO_FUSED_CE=1, off by default).

Upstream's loss path on the bf16 logits (2048 x 100288 per GPU) runs, per update: a cast to fp32, log_softmax in
fp32, nll_loss(sum); backward: a zero-filled fp32 gradient, the nll scatter, the log_softmax backward in fp32 and a cast
back to bf16 -- about 3.8 ms of memory-bound kernels on 0.4-0.8 GB tensors.
Here, for the default language-model path (decoder.get_normalized_probs = fp32 log_softmax of net_output[0], no
adaptive softmax, reduce=True), one Triton kernel reads each bf16 logit row once and computes in fp32 the row's
log-sum-exp (running max and rescaled sum) and loss lse - x[target] (0 for target == padding_idx); the loss is the fp32
sum of the row losses. The backward kernel reads the row again and writes the bf16 gradient
(exp(x - lse) - onehot(target)) * grad directly (zero rows for padding targets), i.e. the same function as
log_softmax/nll backward followed by the bf16 cast, without the fp32 intermediates.
"""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get('YOCO_FUSED_CE', '0') == '1'
_BLOCK = 4096


@triton.jit
def _ce_fwd(X, T, LSE, LOSS, V, stride, ignore, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = X + row * stride
    t = tl.load(T + row)
    cols = tl.arange(0, BLOCK)
    m = tl.max(tl.full([BLOCK], float('-inf'), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for off in range(0, V, BLOCK):
        c = off + cols
        x = tl.load(base + c, mask=c < V, other=float('-inf')).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(s)
    xt = tl.load(base + t).to(tl.float32)
    tl.store(LSE + row, lse)
    tl.store(LOSS + row, tl.where(t == ignore, 0.0, lse - xt))


@triton.jit
def _ce_bwd(X, T, LSE, G, DX, V, stride, ignore, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = X + row * stride
    dbase = DX + row * stride
    t = tl.load(T + row)
    lse = tl.load(LSE + row)
    g = tl.load(G).to(tl.float32)
    g = tl.where(t == ignore, 0.0, g)
    cols = tl.arange(0, BLOCK)
    for off in range(0, V, BLOCK):
        c = off + cols
        mask = c < V
        x = tl.load(base + c, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(c == t, p - 1.0, p)
        tl.store(dbase + c, (p * g).to(DX.dtype.element_ty), mask=mask)


class _FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index):
        N, V = logits.shape
        lse = torch.empty(N, dtype=torch.float32, device=logits.device)
        rows = torch.empty(N, dtype=torch.float32, device=logits.device)
        _ce_fwd[(N,)](logits, target, lse, rows, V, logits.stride(0), ignore_index, BLOCK=_BLOCK, num_warps=8)
        ctx.save_for_backward(logits, target, lse)
        ctx.ignore_index = ignore_index
        return rows.sum()

    @staticmethod
    def backward(ctx, grad):
        logits, target, lse = ctx.saved_tensors
        N, V = logits.shape
        dx = torch.empty_like(logits)
        g = grad.reshape(1).to(torch.float32).contiguous()
        _ce_bwd[(N,)](logits, target, lse, g, dx, V, logits.stride(0), ctx.ignore_index, BLOCK=_BLOCK, num_warps=8)
        return dx, None, None


def _func(m):
    return getattr(m, '__func__', None)


def _plain_lm(model):
    from fairseq.models.fairseq_decoder import FairseqDecoder
    from fairseq.models.fairseq_model import BaseFairseqModel
    dec = getattr(model, 'decoder', None)
    return (dec is not None
            and _func(model.get_normalized_probs) is BaseFairseqModel.get_normalized_probs
            and _func(model.get_normalized_probs_scriptable) is BaseFairseqModel.get_normalized_probs_scriptable
            and _func(model.get_targets) is BaseFairseqModel.get_targets
            and _func(dec.get_normalized_probs) is FairseqDecoder.get_normalized_probs
            and _func(dec.get_normalized_probs_scriptable) is FairseqDecoder.get_normalized_probs_scriptable
            and getattr(dec, 'adaptive_softmax', None) is None and not getattr(dec, 'onnx_trace', False))


if ENABLED:
    from fairseq.criterions.cross_entropy import CrossEntropyCriterion
    _orig_compute_loss = CrossEntropyCriterion.compute_loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        logits = net_output[0] if isinstance(net_output, (tuple, list)) else None
        if (reduce and torch.is_tensor(logits) and logits.is_cuda and logits.dtype == torch.bfloat16
                and logits.dim() == 3 and _plain_lm(model)):
            target = model.get_targets(sample, net_output).view(-1)
            x = logits.reshape(-1, logits.size(-1))
            if x.stride(-1) != 1:
                x = x.contiguous()
            loss = _FusedCE.apply(x, target.contiguous(), int(self.padding_idx))
            return loss, loss
        return _orig_compute_loss(self, model, net_output, sample, reduce)

    CrossEntropyCriterion.compute_loss = compute_loss
