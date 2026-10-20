"""CUDA-graph the YOCO decoder layers for training (switch YOCO_LAYER_GRAPHS=1, off by default).

The decoder layers are launch-bound at 2048 tokens per GPU: in the reference profile the forward kernels start a
median 3 us after their launch (the GPU waits for the CPU) and about half of the backward kernels too. Each
DecoderLayer (13 self layers: gate retention + SwiGLU FFN; 13 cross layers: flash attention on the shared KV + FFN)
is captured once, forward and backward, with torch.cuda.make_graphed_callables; replays run the same kernels on the
same inputs, so the arithmetic is unchanged. Graphs are built at the first training forward with the live shapes;
anything else (eval, no grad, other shapes, incremental decoding) takes the eager path.
"""
import os

import torch
import torch.nn as nn

ENABLED = os.environ.get('YOCO_LAYER_GRAPHS', '0') == '1'
# DDP keeps the gradients as views of its buckets (no bucket -> .grad copy after the all-reduce; same values)
BUCKET_VIEW = os.environ.get('YOCO_DDP_BUCKET_VIEW', '0') == '1'


class _SelfCall(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, cos, sin):
        return self.layer(x, rel_pos=(cos, sin))


class _CrossCall(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, key, value, cos, sin):
        return self.layer(x, key=key, value=value, rel_pos=(cos, sin))


class Graphs:
    @staticmethod
    def signature(*ts):
        return tuple((tuple(t.shape), t.dtype, t.device, t.requires_grad) for t in ts)


def usable(module, incremental_state, x):
    return (ENABLED and module.training and torch.is_grad_enabled() and incremental_state is None and x.is_cuda
            and not torch.cuda.is_current_stream_capturing())


def prepare(model, x, incremental_state, start_pos, skip_cross_decoder):
    """(self-layer calls, cross-layer calls) for this forward, or None for the eager path."""
    st = getattr(model, '_layer_graphs', None)
    if st is None or not usable(model, incremental_state, x) or start_pos != 0 or skip_cross_decoder:
        return None
    return st[1] if st[0] == Graphs.signature(x) else None


def build(model, bsz, seqlen):
    """Capture all 26 layers in one make_graphed_callables call, in runtime order (self 0..12, cross 0..12; backward
    in reverse), one memory pool. Must run before DDP creates the parameters' gradient accumulators: an accumulator
    created on the default stream cannot receive gradients inside a capture ("operation would make the legacy stream
    depend on a capturing blocking stream")."""
    sd, cd = model.self_decoder, model.cross_decoder
    w = model.embed_tokens.weight
    dev, dt, dim = w.device, w.dtype, model.args.dim
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)                           # private generator: the global RNG streams are untouched

    def rnd(*shape, grad):
        t = torch.empty(*shape, device=dev, dtype=dt)
        t.normal_(generator=gen).mul_(0.02)
        return t.requires_grad_(grad)
    samples, mods = [], []
    for layer in sd.layers:
        mods.append(_SelfCall(layer))
        samples.append((rnd(bsz, seqlen, dim, grad=True), rnd(seqlen, sd.head_dim // 2, grad=False),
                        rnd(seqlen, sd.head_dim // 2, grad=False)))
    for layer in cd.layers:
        mods.append(_CrossCall(layer))
        samples.append((rnd(bsz, seqlen, dim, grad=True), rnd(bsz, seqlen, cd.num_heads, cd.head_dim, grad=True),
                        rnd(bsz, seqlen, cd.num_heads, cd.head_dim, grad=True),
                        rnd(seqlen, cd.head_dim // 2, grad=False), rnd(seqlen, cd.head_dim // 2, grad=False)))
    calls = torch.cuda.make_graphed_callables(tuple(mods), tuple(samples))
    n = len(sd.layers)
    sig = Graphs.signature(torch.empty(bsz, seqlen, dim, device=dev, dtype=dt).requires_grad_(True))
    model._layer_graphs = (sig, (calls[:n], calls[n:]))


def _prebuild(module):
    from .yoco import YOCO
    for m in module.modules():
        if isinstance(m, YOCO) and m.training and m.embed_tokens.weight.is_cuda and m.args.sliding_window is None:
            seqlen = m.args.max_seq_len
            if seqlen % m.self_decoder.block_size == 0 and getattr(m, '_layer_graphs', None) is None:
                build(m, max(1, m.args.max_batch_size), seqlen)


# callables run on the module right before DDP wraps it (and before the layers are captured), e.g. zero1_fast's
# placement of the bf16 parameters into its all-gather buffer
PRE_DDP_HOOKS = []


def install_ddp_patch():
    import fairseq.models.distributed_fairseq_model as _dfm
    if getattr(_dfm.DistributedDataParallel, '_yoco_patched', False):
        return

    class _GraphedDDP(_dfm.DistributedDataParallel):
        """torch DDP that first runs the pre-DDP hooks and captures the YOCO layers (see build())."""
        _yoco_patched = True

        def __init__(self, module, *args, **kwargs):
            for hook in PRE_DDP_HOOKS:
                hook(module)
            if ENABLED:
                _prebuild(module)
            if BUCKET_VIEW:
                kwargs['gradient_as_bucket_view'] = True
            super().__init__(module, *args, **kwargs)

    _dfm.DistributedDataParallel = _GraphedDDP


if ENABLED or BUCKET_VIEW:
    install_ddp_patch()
