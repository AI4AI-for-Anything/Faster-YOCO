"""Training-validation harness (identical for every arm): runs <tree>/yoco/train.py exactly as the bench recipe does,
with one change made from outside the tree: fairseq's checkpoint_utils.save_checkpoint is replaced by a model-only
save on data-parallel rank 0 (bf16 model state_dict, no optimizer consolidation) at every save point
(--save-interval-updates and the last update). It also logs each rank's peak memory at every save point.
usage: python train_wrapper.py <tree> <ckpt_dir> <train.py args...>   (run under torchrun)
"""
import os, sys, runpy, time, json
tree, ckpt_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
entry = os.path.join(tree, 'yoco', 'train.py')
os.chdir(os.path.join(tree, 'yoco'))
sys.path.insert(0, os.path.join(tree, 'yoco'))
sys.argv = [entry] + sys.argv[3:]

import torch
from fairseq import checkpoint_utils


def _fail_fast(etype, value, tb):
    # a failing rank must not hang in interpreter shutdown (NCCL, data-loader processes): print and exit
    import traceback
    traceback.print_exception(etype, value, tb)
    sys.stderr.flush(); sys.stdout.flush()
    os._exit(1)


sys.excepthook = _fail_fast

_saved = set()


def save_checkpoint(cfg, trainer, epoch_itr, val_loss, *args, **kwargs):
    n = trainer.get_num_updates()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    mem = {'rank': rank, 'num_updates': n, 'max_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
           'max_reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'time': time.time()}
    print('VALIDATION_MEM ' + json.dumps(mem), flush=True)
    if n in _saved or n == 0:
        return None
    _saved.add(n)
    if trainer.data_parallel_rank == 0:
        t0 = time.time()
        sd = {k: v.detach().to('cpu', copy=True) for k, v in trainer.get_model().state_dict().items()}
        path = os.path.join(ckpt_dir, f'model_{n:06d}.pt')
        torch.save({'model': sd, 'num_updates': n}, path + '.tmp')
        os.replace(path + '.tmp', path)
        print(f'VALIDATION_CKPT {path} {time.time() - t0:.1f}s', flush=True)
    return None


checkpoint_utils.save_checkpoint = save_checkpoint
os.makedirs(ckpt_dir, exist_ok=True)
try:
    runpy.run_path(entry, run_name='__main__')
except SystemExit as e:
    if e.code not in (0, None):
        _fail_fast(*sys.exc_info())
except BaseException:
    _fail_fast(*sys.exc_info())
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
