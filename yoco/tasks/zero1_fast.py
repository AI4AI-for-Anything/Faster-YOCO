"""Sharded fp32 work for fairseq's FP16Optimizer under --zero-sharding os (switch YOCO_ZERO1_FAST=1, off by default).

Upstream (bf16, ZeRO-1 through fairscale OSS), per update, every rank:
  clip:  copies ALL bf16 grads to fp32 master grads, multi_tensor_l2norm over all fp32 grads;
  step:  multiplies ALL fp32 grads by the (sample-size x clip) factor, steps Adam on its own shard, OSS broadcasts the
         full fp32 master weights (13.8 GB), then copies ALL fp32 masters back into the bf16 model;
  zero_grad: memsets ALL fp32 master grads.
Here the same math is done only where it is consumed:
  clip:  the same fairseq routine (multi_tensor_l2norm, fp32 accumulation) on the bf16 grads, whose fp32 upcast is
         exact -> the same norm and clip factor;
  step:  bf16 -> fp32 grad copy and the factor multiply only for this rank's OSS shard, the same Adam step (through
         fp32_optimizer.step), then each owner casts its updated fp32 shard to bf16 (the same rounding as upstream's
         copy) and broadcasts bf16 (6.9 GB) straight into the model. Non-owned fp32 masters / grads are not
         maintained (ZeRO-1 does not need them).
"""
import os

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.optim import fp16_optimizer as _f16

ENABLED = os.environ.get('YOCO_ZERO1_FAST', '0') == '1'
# one in-place all_gather_into_tensor over equal-size (padded) per-owner segments instead of one broadcast per owner
ALLGATHER = os.environ.get('YOCO_ZERO1_ALLGATHER', '0') == '1'
# the bf16 model parameters live in the all-gather buffer itself (placed before DDP and the layer graphs)
PARAM_VIEWS = ALLGATHER and os.environ.get('YOCO_ZERO1_PARAM_VIEWS', '0') == '1'
REDUCE_OWNER = (PARAM_VIEWS and os.environ.get('YOCO_PARAM_ALLREDUCE', '0') == '1'
                and os.environ.get('YOCO_GRAD_REDUCE_OWNER', '0') == '1')
_PLACED = {}

_orig_clip = _f16._FP16OptimizerMixin.clip_grad_norm
_orig_step = _f16._FP16OptimizerMixin.step
_orig_zero = _f16._FP16OptimizerMixin.zero_grad


def _place_params(module):
    """Replicate fairscale OSS's greedy partition (params in order, each to the rank with the smallest total so far)
    and make every trainable bf16 parameter a view of its owner's segment of one flat buffer."""
    if not dist.is_initialized():
        return
    params = [p for p in module.parameters() if p.requires_grad]
    if not params or any(not p.is_cuda for p in params):
        return
    world = dist.get_world_size()
    sizes, owner = [0] * world, []
    for p in params:
        r = sizes.index(min(sizes))
        owner.append(r)
        sizes[r] += p.numel()
    seg = max(sizes)
    flat = torch.empty(world * seg, dtype=params[0].dtype, device=params[0].device)
    offs = [r * seg for r in range(world)]
    with torch.no_grad():
        for p, r in zip(params, owner):
            v = flat[offs[r]:offs[r] + p.numel()].view_as(p)
            v.copy_(p.data)
            p.data = v
            offs[r] += p.numel()
    _PLACED.update(flat=flat, seg=seg, owner={id(p): r for p, r in zip(params, owner)})


def _oss(self):
    inner = getattr(self.fp32_optimizer, 'optimizer', None)
    if inner is None or not hasattr(inner, 'partition_parameters') or not hasattr(inner, 'optim'):
        return None
    return inner


def _setup(self):
    st = getattr(self, '_z1', None)
    if st is not None:
        return st
    oss = _oss(self)
    if oss is None or self.has_flat_params or self.scaler is not None:
        self._z1 = False
        return False
    p32s = list(self.fp32_params)
    p16s = [p for p in self.fp16_params]
    assert len(p32s) == len(p16s)
    owner = oss._param_to_rank
    rank, world, group = oss.rank, oss.world_size, oss.group
    pairs = [(p16, p32) for p16, p32 in zip(p16s, p32s) if p16.requires_grad]
    local = [(p16, p32) for p16, p32 in pairs if owner[p32] == rank]
    # one flat bf16 buffer per owner rank, params in model order; non-owners receive into it
    per = [[(p16, p32) for p16, p32 in pairs if owner[p32] == r] for r in range(world)]
    seg = max(sum(p16.numel() for p16, _ in mine) for mine in per)
    placed = bool(_PLACED) and all(_PLACED['owner'].get(id(p16)) == owner[p32] for p16, p32 in pairs)
    if PARAM_VIEWS and not placed:
        raise RuntimeError('zero1_fast: the placed parameter layout does not match the OSS partition')
    if placed:
        flat, seg = _PLACED['flat'], _PLACED['seg']
    else:
        flat = torch.empty(world * seg, dtype=p16s[0].dtype, device=p16s[0].device) if ALLGATHER else None
    buckets = []
    for r, mine in enumerate(per):
        n = sum(p16.numel() for p16, _ in mine)
        buf = flat[r * seg:r * seg + n] if ALLGATHER else torch.empty(n, dtype=p16s[0].dtype, device=p16s[0].device)
        views, off = [], 0
        for p16, p32 in mine:
            v = buf[off:off + p16.numel()].view_as(p16)
            if placed and v.data_ptr() != p16.data_ptr():
                raise RuntimeError('zero1_fast: a placed parameter is not at its slot')
            views.append((p16, p32, v))
            off += p16.numel()
        buckets.append((r, buf, views))
    oss._broadcast_params = lambda: None          # the fp32 broadcast is replaced by the bf16 one below
    st = {'local': local, 'buckets': buckets, 'rank': rank, 'group': group, 'global_ranks': oss._local_to_global_rank,
          'flat': flat, 'seg': seg, 'placed': placed}
    self._z1 = st
    return st


def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
    st = _setup(self) if ENABLED else False
    if not st:
        return _orig_clip(self, max_norm, aggregate_norm_fn)
    params = [p for p in self.fp16_params if p.requires_grad]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    if REDUCE_OWNER:
        # only the owner holds the reduced gradient: sum the owners' squared norms (fairseq's routine per owner)
        owned = [p16 for p16, _ in st['local']]
        sq = utils.clip_grad_norm_(owned, 0, None).float() ** 2
        dist.all_reduce(sq, group=st['group'])
        norm = sq.sqrt()
        if aggregate_norm_fn is not None:
            norm = aggregate_norm_fn(norm)
        grad_norm = self._multiply_factor * norm
    else:
        grad_norm = self._multiply_factor * utils.clip_grad_norm_(params, 0, aggregate_norm_fn)
    if max_norm > 0.0:
        clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
        self._multiply_factor *= clip_coef
    self._needs_sync = False
    return grad_norm


@torch.no_grad()
def step(self, closure=None, groups=None):
    st = _setup(self) if ENABLED else False
    if not st:
        return _orig_step(self, closure, groups)
    factor = self._multiply_factor
    for p16, p32 in st['local']:
        if p16.grad is None:
            p32.grad.zero_()
        else:
            p32.grad.copy_(p16.grad)
        if torch.is_tensor(factor) or factor != 1.0:
            p32.grad.mul_(factor.to(p32.grad.device) if torch.is_tensor(factor) else factor)
    self._multiply_factor = 1.0
    self._needs_sync = False
    self.fp32_optimizer.step(closure, groups=groups)          # the Adam step on this rank's shard (OSS)
    rank = st['rank']
    for r, buf, views in st['buckets']:
        if r == rank:
            for p16, p32, v in views:
                v.copy_(p32.data)
    if st['flat'] is not None:
        flat, seg = st['flat'], st['seg']
        dist.all_gather_into_tensor(flat, flat[rank * seg:(rank + 1) * seg], group=st['group'])
    else:
        handles = [dist.broadcast(buf, src=st['global_ranks'][r], group=st['group'], async_op=True)
                   for r, buf, views in st['buckets'] if buf.numel()]
        for h in handles:
            h.wait()
    if not st['placed']:
        for r, buf, views in st['buckets']:
            for p16, p32, v in views:
                p16.data.copy_(v)


def zero_grad(self):
    st = _setup(self) if ENABLED else False
    if not st:
        return _orig_zero(self)
    for p in self.fp16_params:
        p.grad = None
    self._needs_sync = False


if ENABLED and PARAM_VIEWS:
    from models.decoder import layer_graphs as _lg
    _lg.PRE_DDP_HOOKS.append(_place_params)
    _lg.install_ddp_patch()

if ENABLED:
    _f16._FP16OptimizerMixin.clip_grad_norm = clip_grad_norm
    _f16._FP16OptimizerMixin.step = step
    _f16._FP16OptimizerMixin.zero_grad = zero_grad
