"""Held-out loss of saved checkpoints, identical code for every arm: the REFERENCE tree's YOCO model (upstream code
at 1d0c1ce, yoco_3b shape, bf16), eval mode, no grad; loss = fp32 log_softmax cross-entropy of the next token, as
fairseq's cross_entropy criterion computes it, averaged over all target tokens (no padding in packed samples).
Samples are split across the 8 ranks; sums are all-reduced in float64.
usage: torchrun --nproc-per-node=8 eval_ckpt.py <ref tree> <heldout.pt> <out.json> <ckpt>..."""
import json, math, os, sys, time
ref, heldout, out_path, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, os.path.join(ref, 'yoco'))
os.chdir(os.path.join(ref, 'yoco'))
import torch
import torch.distributed as dist
import torch.nn.functional as F
dist.init_process_group('nccl')
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
from fairseq.model_parallel.megatron.mpu import initialize_model_parallel, model_parallel_is_initialized
if not model_parallel_is_initialized():          # as models/yoco.py LanguageModel.build_model does
    initialize_model_parallel(1)
from models.decoder.yoco import YOCO, YOCOArgs
H = torch.load(heldout)
args = YOCOArgs(dim=3072, n_layers=26, n_self_heads=24, n_attn_heads=24, n_attn_kv_heads=8, hidden_dim=8192,
                vocab_size=H['n_words'], max_batch_size=1, max_seq_len=H['tokens_per_sample'], model_parallel_size=1,
                load_checkpoint=False, rope_theta=10000.0)
model = YOCO(args).cuda().to(torch.bfloat16).eval()
results = json.load(open(out_path)) if os.path.exists(out_path) else {}
own = model.state_dict()
for ck in ckpts:
    if os.path.basename(ck) in results:
        continue
    t0 = time.time()
    # rank 0 reads the checkpoint (one read of the shared file system), every tensor is broadcast to the others
    meta = [None]
    if rank == 0:
        sd = torch.load(ck, map_location='cpu')      # one sequential read (mmap page faults are slow here)
        st = {k[len('decoder.model.'):] if k.startswith('decoder.model.') else k: v for k, v in sd['model'].items()}
        if sorted(st) != sorted(own) or any(tuple(st[k].shape) != tuple(own[k].shape) for k in own):
            raise SystemExit(f'{ck}: state dict does not match the model')
        meta = [sd['num_updates']]
    dist.broadcast_object_list(meta, src=0)
    with torch.no_grad():
        for k in sorted(own):
            if rank == 0:
                own[k].copy_(st[k])
            dist.broadcast(own[k], src=0)
    r = {'num_updates': meta[0], 'load_s': time.time() - t0}
    for name, data in H['samples'].items():
        tot = torch.zeros(2, dtype=torch.float64, device='cuda')
        with torch.no_grad():
            for i in range(rank, data.shape[0], world):
                s = data[i].cuda()
                logits, _ = model(s[None, :-1])
                lp = F.log_softmax(logits.float(), dim=-1)
                nll = F.nll_loss(lp.view(-1, lp.shape[-1]), s[1:], ignore_index=H['pad_id'], reduction='sum')
                tot += torch.stack([nll.double(), (s[1:] != H['pad_id']).sum().double()])
        dist.all_reduce(tot)
        r[name] = {'loss_nats': float(tot[0] / tot[1]), 'loss_bits': float(tot[0] / tot[1] / math.log(2)),
                   'tokens': int(tot[1])}
    r['mix_0.8_0.2_bits'] = 0.8 * r['fineweb_edu']['loss_bits'] + 0.2 * r['codeparrot']['loss_bits']
    r['eval_s'] = time.time() - t0
    results[os.path.basename(ck)] = r
    if rank == 0:
        print('EVAL', os.path.basename(ck), json.dumps(r), flush=True)
        json.dump(results, open(out_path, 'w'), indent=1)
dist.destroy_process_group()
