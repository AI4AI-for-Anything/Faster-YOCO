import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.model_parallel.megatron.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from .rms_norm import RMSNorm

from .kernel.gate_recurrent import chunk_gate_retention, recurrent_gate_retention
from .kernel.rotary import apply_rotary_emb
from .kernel.swiglu import swiglu

from .model_parallel_init import qkvg_init_method, out_init_method

import os
# switch YOCO_SUBLN_FUSED=1: the weightless per-head subln and the head -> token transpose after it in one Triton
# kernel each way (rms_triton.rms_norm_heads; same arithmetic as the Triton RMSNorm)
_SUBLN_FUSED = os.environ.get('YOCO_SUBLN_FUSED', '0') == '1'
# switch YOCO_FUSE_QKVG=1: the q/k/v/g projections of the training forward as one GEMM against the four weights
# concatenated (copied each forward); forward outputs are column blocks of it, the backward computes one input
# gradient GEMM (fp32 accumulation over the four instead of three bf16 adds) and one weight-gradient GEMM whose
# row blocks are the four weight gradients
_FUSE_QKVG = os.environ.get('YOCO_FUSE_QKVG', '0') == '1'

class GateRetention(nn.Module):

    def __init__(
        self,
        args,
        gate_logit_normalizer: int = 16,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.dim
        self.num_heads = args.n_self_heads // args.model_parallel_size
        self.head_dim = args.dim // args.n_self_heads

        self.q_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.k_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.v_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.g_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.gt_proj = ColumnParallelLinear(args.dim, args.n_self_heads, bias=False, gather_output=False, init_method=qkvg_init_method)
        
        self.out_proj = RowParallelLinear(args.dim, args.dim, bias=False, input_is_parallel=True, init_method=out_init_method)

        self.subln = RMSNorm(self.head_dim, elementwise_affine=False, eps=args.norm_eps)

        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        x,
        rel_pos,
        incremental_state=None,
        is_prefilling=False,
    ):
        bsz, tgt_len, _ = x.size()

        if (_FUSE_QKVG and self.training and torch.is_grad_enabled() and incremental_state is None and x.is_cuda
                and self.args.model_parallel_size == 1
                and all(m.bias is None for m in (self.q_proj, self.k_proj, self.v_proj, self.g_proj))):
            w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.g_proj.weight], 0)
            q, k, v, g = F.linear(x, w).split([self.q_proj.weight.shape[0], self.k_proj.weight.shape[0],
                                               self.v_proj.weight.shape[0], self.g_proj.weight.shape[0]], dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            g = self.g_proj(x)
        gt = self.gt_proj(x)

        qr = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        kr = k.view(bsz, tgt_len, self.num_heads, self.head_dim)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        gt = gt.view(bsz, tgt_len, self.num_heads).transpose(1, 2)

        qr = apply_rotary_emb(qr, *rel_pos, interleaved=True).transpose(1, 2)
        kr = apply_rotary_emb(kr, *rel_pos, interleaved=True).transpose(1, 2)
        gt = (F.logsigmoid(gt) / self.gate_logit_normalizer)

        if incremental_state is not None and not is_prefilling:
            o = recurrent_gate_retention(qr, kr, v, gt, incremental_state)
        else:
            if incremental_state is not None:
                index_mask = incremental_state["index_mask"]
                gt_sum = gt.float().masked_fill(index_mask, 0).sum(dim=-1, keepdim=True)
                gt_mask = (gt_sum - gt.float().cumsum(dim=-1)).exp().masked_fill(index_mask, 0)
                next_hidden_state = (kr.transpose(-1, -2) * (self.head_dim ** -0.5)) @ (v * gt_mask.to(v.dtype).unsqueeze(-1))
                if "last_hidden_state" in incremental_state:
                    last_hidden_state = incremental_state["last_hidden_state"]
                    next_hidden_state += last_hidden_state * gt_sum.exp().unsqueeze(-1).to(v.dtype) if last_hidden_state is not None else 0
                else:
                    last_hidden_state = None
                incremental_state["last_hidden_state"] = next_hidden_state
                o = chunk_gate_retention(qr, kr, v, gt, chunk_size=256, last_hidden_state=last_hidden_state)
            else:
                o = chunk_gate_retention(qr, kr, v, gt, chunk_size=256)
                
        if (_SUBLN_FUSED and o.is_cuda and o.dtype == torch.bfloat16 and self.subln.weight is None
                and o.dim() == 4 and o.shape[-1] <= 256):
            from .rms_triton import rms_norm_heads
            o = rms_norm_heads(o, self.subln.eps)
        else:
            o = self.subln(o).transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        o = swiglu(g, o)
        o = self.out_proj(o)
        return o
