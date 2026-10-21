"""fairseq Trainer.train_step without per-update host syncs (switch YOCO_TRAINER_NOSYNC=1, off by default).

Upstream's train_step makes the CPU wait for the GPU several times per update: `sample_size or 1.0` (truthiness of a
GPU tensor), the cross-rank grad-norm consistency check and `torch.isfinite(grad_norm)` before the optimizer step,
and `torch.isfinite(grad_norm)` plus agg.get_smoothed_values() (which rounds every logged value to a Python float)
after it. Each wait drains the GPU queue, and the GPU then idles while the CPU prepares the next work.

This version runs the same computation for this recipe's path (no TPU, MoE, BMUF, fp16 loss scaling):
- the gradient normalizer is `numer / where(sample_size > 0, sample_size, 1)` (identical values, no sync);
- the two abort checks (non-finite grad norm; grad norms inconsistent across ranks) are computed on the GPU, copied
  to pinned host memory without blocking, and raised as the same FloatingPointError as soon as the copy has landed
  (at the latest one update later, before that update's optimizer step);
- the logged meters receive the same tensors; the per-update dict returned to the loop is not rounded to floats
  (the loop only tests it for None; the smoothed values are computed at each log interval as before);
- the batch is copied to the GPU from pinned memory without blocking (upstream's .to(device, non_blocking=True) on a
  pageable tensor synchronizes the stream, i.e. waits for the whole previous update).
"""
import contextlib
import os

import torch

from fairseq import utils
from fairseq.logging import metrics
from fairseq import trainer as _trainer_mod

ENABLED = os.environ.get('YOCO_TRAINER_NOSYNC', '0') == '1'


class _Pending:
    def __init__(self):
        self.host = torch.zeros(2, dtype=torch.bool, pin_memory=True)
        self.event = None
        self.armed = False

    def arm(self, finite, consistent):
        flags = torch.stack([finite, consistent])
        self.host.copy_(flags, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record()
        self.armed = True

    def check(self, block):
        if not self.armed:
            return
        if not block and not self.event.query():
            return
        self.event.synchronize()
        self.armed = False
        finite, consistent = bool(self.host[0]), bool(self.host[1])
        if not consistent:
            raise FloatingPointError('Fatal error: gradients are inconsistent between workers.')
        if not finite:
            raise FloatingPointError('gradients are Nan/Inf')


def _prepare_sample(self, sample):
    """Trainer._prepare_sample for this path, with a non-blocking host->device copy from pinned memory."""
    if sample == "DUMMY" or sample is None or len(sample) == 0:
        return self._prepare_sample_orig(sample)
    if self.cuda and not self.pipeline_model_parallel:
        dev = torch.cuda.current_device()

        def to_dev(t):
            if t.device.type == 'cpu':
                return t.pin_memory().to(device=dev, non_blocking=True)
            return t.to(device=dev, non_blocking=True)
        sample = utils.apply_to_sample(to_dev, sample)
    else:
        return self._prepare_sample_orig(sample)
    if self.cfg.common.fp16:
        sample = utils.apply_to_sample(lambda t: t.half() if t.dtype is torch.float32 else t, sample)
    if self.cfg.common.bf16:
        sample = utils.apply_to_sample(lambda t: t.to(dtype=torch.bfloat16) if t.dtype is torch.float32 else t,
                                       sample)
    if self._dummy_batch == "DUMMY":
        self._dummy_batch = sample
    return sample, False


def _aggregate(self, logging_outputs, *extra_stats_to_sum, ignore=False):
    """Trainer._fast_stat_sync_sum / distributed_utils.all_reduce_dict with the host values copied from pinned
    memory without blocking (same keys, same order, same double-precision all-reduces, same results)."""
    if not self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
        return self._aggregate_logging_outputs(logging_outputs, *extra_stats_to_sum, ignore=ignore)
    data = {}
    for i, stat in enumerate(extra_stats_to_sum):
        data["extra_stats_" + str(i)] = stat
    if len(logging_outputs) > 0:
        log_keys = list(logging_outputs[0].keys())
        for k in log_keys:
            if not ignore:
                v = sum(log[k] for log in logging_outputs if k in log)
            else:
                v = logging_outputs[0][k]
                v = torch.zeros_like(v) if torch.is_tensor(v) else 0
            data["logging_outputs_" + k] = v
    else:
        log_keys = None
    device, group = self.device, self.data_parallel_process_group
    cpu_keys, dev_keys = [], []
    for k, t in data.items():
        (dev_keys if torch.is_tensor(t) and t.device.type == device.type else cpu_keys).append(k)
    out = {}
    if cpu_keys:
        host = torch.tensor([float(data[k]) if not torch.is_tensor(data[k]) else float(data[k].item())
                             for k in cpu_keys], dtype=torch.double).pin_memory()
        buf = host.to(device=device, non_blocking=True)
        _trainer_mod.distributed_utils.all_reduce(buf, group=group)
        for k, v in zip(cpu_keys, buf):
            out[k] = v
    if dev_keys:
        vals = [data[k].to(dtype=torch.double) for k in dev_keys]
        buf = torch.cat([t.view(-1) for t in vals])
        _trainer_mod.distributed_utils.all_reduce(buf, group=group)
        for k, v, orig in zip(dev_keys, torch.split(buf, [t.numel() for t in vals]), vals):
            out[k] = v.view_as(orig)
    extra = [out["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))]
    logging_outputs = [{k: out["logging_outputs_" + k] for k in log_keys}] if log_keys is not None else []
    return logging_outputs, extra


def _consistent_flag(self, grad_norm):
    """The same test as Trainer._check_grad_norms, as a GPU bool (no host sync)."""
    if self._grad_norm_buf is None:
        return torch.ones((), dtype=torch.bool, device=grad_norm.device)
    self._grad_norm_buf.zero_()
    self._grad_norm_buf[self.data_parallel_rank] = grad_norm
    _trainer_mod.distributed_utils.all_reduce(self._grad_norm_buf, group=self.data_parallel_process_group)
    t = self._grad_norm_buf
    max_abs_diff = torch.max(torch.abs(t - t[0]))
    return torch.isfinite(t).all() & (max_abs_diff / (t[0] + 1e-6) < 1e-6).all()


def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
    if grad_norm is not None:
        metrics.log_speed("ups", 1.0, priority=100, round=2)
        metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
        if self.cfg.optimization.clip_norm > 0:
            metrics.log_scalar("clip", torch.where(grad_norm > self.cfg.optimization.clip_norm,
                                                   grad_norm.new_tensor(100), grad_norm.new_tensor(0)),
                               priority=500, round=1)
    with metrics.aggregate():
        if logging_outputs is not None:
            self.task.reduce_metrics(logging_outputs, self.get_criterion())
            del logging_outputs
    return {"sample_size": sample_size}


@metrics.aggregate("train")
def train_step(self, samples, raise_oom=False):
    """Do forward, backward and parameter update (see the module docstring for what differs from upstream)."""
    pending = getattr(self, '_yoco_pending', None)
    if pending is None:
        pending = self._yoco_pending = _Pending()
    pending.check(block=False)
    self._set_seed()
    self.model.train()
    self.criterion.train()
    self.zero_grad()

    metrics.log_start_time("train_wall", priority=800, round=0)

    logging_outputs, sample_size, ooms = [], 0, 0
    for i, sample in enumerate(samples):
        sample, is_dummy_batch = _prepare_sample(self, sample)

        def maybe_no_sync():
            if self.data_parallel_world_size > 1 and hasattr(self.model, "no_sync") and i < len(samples) - 1:
                return self.model.no_sync()
            return contextlib.ExitStack()

        with maybe_no_sync():
            loss, sample_size_i, logging_output = self.task.train_step(
                sample=sample, model=self.model, criterion=self.criterion, optimizer=self.optimizer,
                update_num=self.get_num_updates(), ignore_grad=is_dummy_batch)
            del loss
        logging_outputs.append(logging_output)
        sample_size += sample_size_i
        if self.cuda and self.get_num_updates() == 0:
            torch.cuda.empty_cache()

    if is_dummy_batch:
        if torch.is_tensor(sample_size):
            sample_size.zero_()
        else:
            sample_size *= 0.0
    if torch.is_tensor(sample_size):
        sample_size = sample_size.float()
    else:
        sample_size = float(sample_size)

    if self._sync_stats():
        train_time = self._local_cumulative_training_time()
        logging_outputs, (sample_size, ooms, total_train_time) = _aggregate(
            self, logging_outputs, sample_size, ooms, train_time, ignore=is_dummy_batch)
        self._cumulative_training_time = total_train_time / self.data_parallel_world_size

    with torch.autograd.profiler.record_function("reduce-grads"):
        self.optimizer.all_reduce_grads(self.model)
        if utils.has_parameters(self.criterion):
            self.optimizer.all_reduce_grads(self.criterion)

    with torch.autograd.profiler.record_function("multiply-grads"):
        numer = self.data_parallel_world_size if not self.cfg.optimization.use_bmuf or self._sync_stats() else 1
        if torch.is_tensor(sample_size):
            denom = torch.where(sample_size > 0, sample_size, torch.ones_like(sample_size))
        else:
            denom = sample_size or 1.0
        self.optimizer.multiply_grads(numer / denom)

    with torch.autograd.profiler.record_function("clip-grads"):
        grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)

    # the update before this one must have passed both abort checks before this update's optimizer step
    pending.check(block=True)
    pending.arm(torch.isfinite(grad_norm).all(), _consistent_flag(self, grad_norm))

    with torch.autograd.profiler.record_function("optimizer"):
        self.task.optimizer_step(self.optimizer, model=self.model, update_num=self.get_num_updates())

    if hasattr(self.model, "perform_additional_optimizer_actions"):
        if hasattr(self.optimizer, "fp32_params"):
            self.model.perform_additional_optimizer_actions(self.optimizer.optimizer, self.optimizer.fp32_params)
        else:
            self.model.perform_additional_optimizer_actions(self.optimizer.optimizer)

    self.set_num_updates(self.get_num_updates() + 1)
    if self.cuda and self.cuda_env is not None:
        self._log_gpu_mem_stats()
    logging_output = _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm)
    if (self.cuda and self.cfg.common.empty_cache_freq > 0 and
            ((self.get_num_updates() + self.cfg.common.empty_cache_freq - 1) % self.cfg.common.empty_cache_freq) == 0):
        torch.cuda.empty_cache()
    metrics.log_stop_time("train_wall")
    return logging_output


if ENABLED:
    _trainer_mod.Trainer._prepare_sample_orig = _trainer_mod.Trainer._prepare_sample
    _trainer_mod.Trainer.train_step = train_step
