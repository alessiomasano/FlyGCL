"""E9c — cost accounting on one GPU: s/batch (median of 50 after 10 warm-up), GFLOPs per online_step and per test image,
peak memory, persistent state (buffers + frozen extra params), for several methods under the standard Si-Blurry config.

python tools/cost_profile.py --methods cflora ranpac flyprompt dualprompt misa mvp codaprompt l2p --out results/cost/cifar100.json
"""
import argparse, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configuration import config as cfgmod
from methods import METHODS
from torch.utils.data import DataLoader
from utils.onlinesampler import OnlineTestSampler
from torch.utils.flop_counter import FlopCounterMode

BASE = ["--dataset", "cifar100", "--data_dir", "./data/CIFAR", "--backbone", "vit_base_patch16_224", "--n_tasks", "5", "--n", "50", "--m", "10",
        "--batchsize", "64", "--online_iter", "3", "--num_epochs", "1", "--use_amp", "--eval_period", "1000", "--n_worker", "4",
        "--transforms", "autoaug", "--seeds", "1", "--log_path", "./results_cost"]
PER_METHOD = {
    "cflora": ["--method", "cflora", "--lr", "0.0001", "--cf_head", "slda", "--cf_warm", "10000", "--cf_warm_lr", "0.001", "--cf_after", "freeze"],
    "ranpac": ["--method", "ranpac", "--lr", "0.005"], "flyprompt": ["--method", "flyprompt", "--lr", "0.005"],
    "dualprompt": ["--method", "dualprompt", "--lr", "0.005"], "misa": ["--method", "dualprompt", "--lr", "0.005", "--load_pt"],
    "mvp": ["--method", "mvp", "--lr", "0.005"], "codaprompt": ["--method", "codaprompt", "--lr", "0.005"], "l2p": ["--method", "l2p", "--lr", "0.005"],
    "singleprompt": ["--method", "singleprompt", "--lr", "0.005"],
    # final cflora configuration (E16c/E21): adapter dim 32 x 5 blocks, auto stop with delta = 0.02; profiled in the plastic phase and after the freeze
    "cflora_final": ["--method", "cflora", "--lr", "0", "--cf_head", "slda", "--cf_adapt", "adapter", "--cf_adapter_dim", "32", "--cf_adapter_blocks", "5",
                     "--cf_after", "freeze", "--cf_warm_auto", "--cf_auto_patience", "100", "--cf_auto_scale", "total", "--cf_auto_tol", "1.0",
                     "--cf_auto_patience_per_class", "0.5", "--cf_auto_min", "1000", "--cf_auto_margin", "0.02"],
}


def build(name):
    argv = sys.argv; sys.argv = ["x"] + BASE + PER_METHOD[name] + ["--note", f"cost_{name}"]
    args = cfgmod.base_parser() if hasattr(cfgmod, "base_parser") else cfgmod.get_args() if hasattr(cfgmod, "get_args") else None
    sys.argv = argv
    setattr(args, "rnd_seed", 1)
    tr = METHODS[args.method](**vars(args))
    tr.gpu = 0; tr.device = torch.device(0)
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    tr.setup_distributed_dataset(); tr.total_samples = len(tr.train_dataset); tr._init_internal_step_scheduler()
    tr.setup_distributed_model(); tr.train_sampler.set_task(0); tr.online_before_task(0)
    return tr


def state_mb(tr):
    m = tr.model_without_ddp
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_frozen_extra = sum(p.numel() for n, p in m.named_parameters() if not p.requires_grad and not n.startswith(("backbone.blocks", "backbone.patch_embed", "backbone.pos_embed", "backbone.cls_token", "backbone.norm", "backbone.fc_norm", "backbone.head", "backbone.fc")))
    n_buf = sum(b.numel() for n, b in m.named_buffers() if "backbone" not in n or "W_rand" in n)
    head = getattr(tr, "head", None); n_head = 0
    if head is not None:
        for k in ("s", "n", "S", "Q", "G", "W_rand"):
            t = getattr(head, k, None); n_head += t.numel() if torch.is_tensor(t) else 0
    return {"trainable_params": n_train, "extra_frozen_params": n_frozen_extra, "buffers": n_buf, "closed_form_state": n_head,
            "state_MB": 4 * (n_train + n_frozen_extra + n_buf + n_head) / 1e6}


def profile(name, n_batches=60):
    tr = build(name); out = {"method": name, **state_mb(tr)}
    it = iter(tr.train_dataloader); times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_batches):
        x, y, idx = next(it)
        torch.cuda.synchronize(); t0 = time.perf_counter(); tr.online_step(x, y, idx); torch.cuda.synchronize()
        if i >= 10: times.append(time.perf_counter() - t0)
    out["s_per_batch_train"] = float(np.median(times)); out["peak_mem_train_GB"] = torch.cuda.max_memory_allocated() / 1e9
    x, y, idx = next(it)
    try:
        with FlopCounterMode(display=False) as fc:
            tr.online_step(x, y, idx)
        out["GFLOPs_per_online_step"] = fc.get_total_flops() / 1e9
    except Exception as e:
        out["GFLOPs_per_online_step"] = f"n/a ({type(e).__name__})"
    # evaluation on 2000 test images
    sampler = OnlineTestSampler(tr.test_dataset, tr.exposed_classes)
    sub_idx = list(iter(sampler))[:2000]
    loader = DataLoader(torch.utils.data.Subset(tr.test_dataset, sub_idx), batch_size=128, num_workers=4)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        tr.online_evaluate(loader)
    torch.cuda.synchronize(); out["s_per_1k_test_images"] = (time.perf_counter() - t0) / 2
    xb, yb = next(iter(loader))
    try:
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                tr.online_evaluate(DataLoader(torch.utils.data.Subset(tr.test_dataset, sub_idx[:128]), batch_size=128, num_workers=0))
        out["GFLOPs_per_test_image"] = fc.get_total_flops() / 1e9 / 128
    except Exception as e:
        out["GFLOPs_per_test_image"] = f"n/a ({type(e).__name__})"
    # cflora: also the post-warm frozen phase
    if name.startswith("cflora"):
        tr.in_warm = False; tr._set_phase_optimizer(); times = []
        for i in range(30):
            x, y, idx = next(it); torch.cuda.synchronize(); t0 = time.perf_counter(); tr.online_step(x, y, idx); torch.cuda.synchronize()
            if i >= 5: times.append(time.perf_counter() - t0)
        out["s_per_batch_train_postwarm"] = float(np.median(times))
        x, y, idx = next(it)
        with FlopCounterMode(display=False) as fc:
            tr.online_step(x, y, idx)
        out["GFLOPs_per_online_step_postwarm"] = fc.get_total_flops() / 1e9
    print(json.dumps(out, indent=1), flush=True)
    del tr; torch.cuda.empty_cache(); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", nargs="+", default=list(PER_METHOD)); ap.add_argument("--out", default="results/cost/cifar100.json")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True); res = []
    for mth in a.methods:
        try:
            res.append(profile(mth))
        except Exception as e:
            import traceback; traceback.print_exc(); res.append({"method": mth, "error": repr(e)})
        json.dump(res, open(a.out, "w"), indent=1)
