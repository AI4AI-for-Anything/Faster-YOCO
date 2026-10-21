"""Gradient reduction to the owner through the GPUs' copy engines (switch YOCO_CE_REDUCE=1, used by layer_graphs with
YOCO_PARAM_ALLREDUCE + YOCO_GRAD_REDUCE_OWNER + YOCO_ZERO1_PARAM_VIEWS).

With NCCL, the per-parameter reductions of the backward run as 24-CTA kernels on the SMs next to the backward GEMMs;
profiled without that overlap the backward GEMMs take ~10 ms less per update. Here no SM work runs during the
backward for the reduction: each owner rank r has a staging buffer [world-1][G_r] (G_r = the elements of the
gradients whose optimizer state it owns, in model order) and an output buffer [G_r]; as soon as a gradient is
accumulated, its rank copies it with cudaMemcpyAsync on a copy stream into its slot of the owner's staging buffer (a
peer write over NVLink through the copy engine, CUDA IPC mapping opened once) or, for its own parameters, into its
output buffer, and drops its .grad (the copy keeps the block alive through record_stream). After the backward, a 1-element NCCL all-reduce enqueued on every rank's copy
stream acts as a GPU-side barrier (it can only complete once every rank has run it, i.e. after every rank's copies),
then the owner sums the 8 contributions in fp32 in rank order, scales by 1/world (AVG) and rounds once to bf16
into the output buffer; its parameters' .grad become views of it. Non-owners keep no gradient (never used under
reduce-to-owner). The staging slots of update k+1 are written only after the owner's sum of update k: a rank's
backward k+1 follows its forward k+1, which follows the bf16 all-gather of update k, in which the owner takes part
after its sum and Adam step.
"""
import ctypes
import glob
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

ENABLED = os.environ.get('YOCO_CE_REDUCE', '0') == '1'


class _IpcHandle(ctypes.Structure):
    _fields_ = [('reserved', ctypes.c_char * 64)]


_RT = None


def _rt():
    global _RT
    if _RT is None:
        import nvidia.cuda_runtime
        base = os.path.dirname(nvidia.cuda_runtime.__file__)
        rt = ctypes.CDLL(sorted(glob.glob(os.path.join(base, 'lib', 'libcudart.so*')))[0])
        rt.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
        rt.cudaIpcOpenMemHandle.restype = ctypes.c_int
        rt.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                       ctypes.c_void_p]
        rt.cudaMemcpyAsync.restype = ctypes.c_int
        rt.cudaGetErrorString.argtypes = [ctypes.c_int]
        rt.cudaGetErrorString.restype = ctypes.c_char_p
        _RT = rt
    return _RT


def _check(err, what):
    if err != 0:
        raise RuntimeError(f'ce_reduce: {what} failed: {_rt().cudaGetErrorString(err).decode()}')


@triton.jit
def _sum_slots(S, OUT, G, R, scale, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    # OUT holds this rank's own gradients, S the WORLD-1 peer slots (peer s < R in slot s, s > R in slot s-1);
    # summed in fp32 in rank order 0..WORLD-1, scaled, rounded once, written back to OUT
    G64 = G.to(tl.int64)                 # (WORLD-1) * G exceeds 2^31 elements: 64-bit slot offsets
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < G64
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(WORLD):
        if k < R:
            acc += tl.load(S + k * G64 + offs, mask=mask, other=0.0).to(tl.float32)
        elif k == R:
            acc += tl.load(OUT + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            acc += tl.load(S + (k - 1) * G64 + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + offs, (acc * scale).to(OUT.dtype.element_ty), mask=mask)


def peer_pointers(t, group):
    """Device addresses, valid in this process, of every rank's tensor t (same call on all ranks): own data_ptr, and
    the peers' buffers mapped through CUDA IPC (opened once, on this process's device, peer access enabled)."""
    st = t.untyped_storage()._share_cuda_()
    handle, off = st[1], int(st[3])                       # IPC handle of the allocation block, byte offset in it
    if not isinstance(handle, bytes) or len(handle) != 64:
        raise RuntimeError(f'ce_reduce: unexpected _share_cuda_() result {type(handle)}')
    off += t.storage_offset() * t.element_size()
    infos = [None] * dist.get_world_size(group)
    dist.all_gather_object(infos, (handle, off), group=group)
    me = dist.get_rank(group)
    out = []
    for r, (h_bytes, o) in enumerate(infos):
        if r == me:
            out.append(t.data_ptr())
            continue
        ptr = ctypes.c_void_p()
        h = _IpcHandle()
        ctypes.memmove(ctypes.addressof(h), h_bytes, 64)
        _check(_rt().cudaIpcOpenMemHandle(ctypes.byref(ptr), h, 1), 'cudaIpcOpenMemHandle')
        out.append(ptr.value + o)
    return out


def memcpy_async(dst, src, nbytes, stream):
    """cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault, stream) on raw device addresses (copy engine)."""
    _check(_rt().cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes), 4,
                                 ctypes.c_void_p(stream.cuda_stream)), 'cudaMemcpyAsync')


class CEReduce:
    """Per-DDP-module state. params: the trainable parameters reduced this way (model order); owner: id -> rank."""

    def __init__(self, params, owner, group):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        dev = params[0].device
        dtype = params[0].dtype
        self.dtype, self.esize = dtype, params[0].element_size()
        sizes = [0] * self.world
        self.where = {}                                    # id(p) -> (owner, element offset in the owner's G)
        for p in params:
            r = owner[id(p)]
            self.where[id(p)] = (r, sizes[r])
            sizes[r] += p.numel()
        self.sizes = sizes
        G = sizes[self.rank]
        self.G = G
        self.mine = [p for p in params if owner[id(p)] == self.rank]
        self.staging = torch.empty((self.world - 1) * max(G, 1), dtype=dtype, device=dev)
        self.out = torch.empty(max(G, 1), dtype=dtype, device=dev)
        self.base = peer_pointers(self.staging, group)
        self.stream = torch.cuda.Stream(device=dev)
        self.flag = torch.zeros(1, dtype=torch.float32, device=dev)
        self.pushed = 0

    def push(self, p):
        r, off = self.where[id(p)]
        g = p.grad
        if not g.is_contiguous():
            raise RuntimeError('ce_reduce: non-contiguous gradient')
        if r == self.rank:
            dst = self.out.data_ptr() + off * self.esize
        else:
            slot = self.rank - (1 if self.rank > r else 0)
            dst = self.base[r] + (slot * self.sizes[r] + off) * self.esize
        cur = torch.cuda.current_stream(g.device)
        self.stream.wait_stream(cur)
        memcpy_async(dst, g.data_ptr(), g.numel() * self.esize, self.stream)
        g.record_stream(self.stream)
        p.grad = None          # copied (the allocator keeps the block until the copy stream is past the copy)
        self.pushed += 1

    def finish(self):
        """After the backward: barrier on the copy streams, sum the owner's slots, point .grad at the result."""
        if self.pushed != len(self.where):
            raise RuntimeError(f'ce_reduce: {self.pushed} of {len(self.where)} gradients were produced this update')
        self.pushed = 0
        with torch.cuda.stream(self.stream):
            work = dist.all_reduce(self.flag, group=self.group, async_op=True)
        work.wait()
        G = self.G
        if G:
            BLOCK = 2048
            _sum_slots[(triton.cdiv(G, BLOCK),)](self.staging, self.out, G, self.rank, 1.0 / self.world,
                                                  WORLD=self.world, BLOCK=BLOCK, num_warps=8)
        off = 0
        for p in self.mine:
            p.grad = self.out[off:off + p.numel()].view_as(p)
            off += p.numel()
