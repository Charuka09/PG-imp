"""
pg_pruning_cifar100.py – PG-based structured pruning on GPU for all CIFAR-100 models.

GPU upgrades vs the previous version:
  • FC fine-tuning uses AMP (FP16)  via pruning_utils.fine_tune_fc
  • DataLoaders use pin_memory + non_blocking
  • importance computation uses torch GPU matmul instead of numpy
  • torch.cuda.empty_cache() after each pruning run

PCA (sklearn) runs on CPU — it receives at most (100, 512) data,
so GPU overhead would outweigh any gain.
"""

import copy, itertools, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (TRAINED_MODELS_DIR, RESULTS_DIR, DATA_DIR, PG_DIR,
                    CIFAR100_MEAN, CIFAR100_STD, NUM_CLASSES, PG_FC_LAYER)
from models import Conv2Net, Conv6Net, VGG16CIFAR
from pruning_utils import (prune_conv, prune_bn, update_next_conv_in,
                            rebuild_fc, eval_acc, compute_ece, fine_tune_fc)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
PCA_COMPS  = 8
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
PIN_MEMORY = (device.type == "cuda")

train_loader = DataLoader(
    datasets.CIFAR100(DATA_DIR, train=True,  download=True,
                      transform=transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomCrop(32, padding=4),
                          transforms.ColorJitter(0.2,0.2,0.2,0.1),
                          transforms.RandomRotation(15),
                          transforms.ToTensor(),
                          transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                      ])),
    batch_size=BATCH, shuffle=True, num_workers=4,
    pin_memory=PIN_MEMORY, persistent_workers=True)
val_loader = DataLoader(
    datasets.CIFAR100(DATA_DIR, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                      ])),
    batch_size=256, shuffle=False, num_workers=4,
    pin_memory=PIN_MEMORY, persistent_workers=True)

VGG_SEQ = [
    ("conv1_1","bn1_1","conv1_2"), ("conv1_2","bn1_2","conv2_1"),
    ("conv2_1","bn2_1","conv2_2"), ("conv2_2","bn2_2","conv3_1"),
    ("conv3_1","bn3_1","conv3_2"), ("conv3_2","bn3_2","conv3_3"),
    ("conv3_3","bn3_3","conv4_1"), ("conv4_1","bn4_1","conv4_2"),
    ("conv4_2","bn4_2","conv4_3"), ("conv4_3","bn4_3","conv5_1"),
    ("conv5_1","bn5_1","conv5_2"), ("conv5_2","bn5_2","conv5_3"),
    ("conv5_3","bn5_3", None),
]
CONV6_SEQ = [
    ("conv1","bn1","conv2"), ("conv2","bn2","conv3"),
    ("conv3","bn3","conv4"), ("conv4","bn4","conv5"),
    ("conv5","bn5","conv6"), ("conv6","bn6", None),
]
CONV2_SEQ = [("conv1","bn1","conv2"), ("conv2","bn2",None)]
SEQ = {"conv2net": CONV2_SEQ, "conv6net": CONV6_SEQ, "vgg16": VGG_SEQ}

MODEL_SPECS = {
    "conv2net": (Conv2Net,    "fc",  ["fc"],         10),
    "conv6net": (Conv6Net,    "fc1", ["fc1","fc2"],  10),
    "vgg16":    (VGG16CIFAR,  "fc1", ["fc1","fc2"],  15),
}


# ── GPU importance scoring ────────────────────────────────────

def filter_importance_gpu(conv_flat_np, fc_w_np, components_np):
    """
    Compute per-filter importance using GPU tensor matmul.

    conv_flat  : (out_ch, kH*kW*in_ch)
    fc_w       : (out_fc, in_fc)
    components : (n_comp, units)
    """
    conv  = torch.tensor(conv_flat_np, dtype=torch.float32, device=device)  # (F, D)
    avg_fc = torch.tensor(fc_w_np.mean(axis=0), dtype=torch.float32, device=device)  # (U,)
    comps  = torch.tensor(components_np, dtype=torch.float32, device=device)  # (C, U)

    scores = []
    for comp in comps:
        sz   = min(conv.shape[1], avg_fc.shape[0], comp.shape[0])
        proj = conv[:, :sz] @ (comp[:sz] * avg_fc[:sz])   # (F,)
        scores.append(proj)

    imp = torch.stack(scores).abs().mean(dim=0).cpu().numpy()  # (F,)
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
    return imp


# ── PG data loading ───────────────────────────────────────────

def load_pg_diffs(model_name, fc_layer):
    pg_base = Path(PG_DIR) / model_name
    diffs   = []
    for cls in range(NUM_CLASSES):
        pc = pg_base / f"correct/pg1_data/{cls}_{fc_layer}_pg1.npy"
        pi = pg_base / f"incorrect/pg1_data/{cls}_{fc_layer}_pg1.npy"
        if pc.exists() and pi.exists():
            diffs.append(np.load(pc) - np.load(pi))
    return np.stack(diffs) if diffs else None


def apply_pruning(m, conv_seq, importances, threshold):
    kept = {}
    for i, (ca, ba, na) in enumerate(conv_seq):
        conv = getattr(m, ca)
        mask = importances[i] > threshold
        nc, idx = prune_conv(conv, mask, device=str(device))
        nb      = prune_bn(getattr(m, ba), idx, device=str(device))
        setattr(m, ca, nc); setattr(m, ba, nb)
        kept[ca] = len(idx)
        if na:
            setattr(m, na, update_next_conv_in(idx, getattr(m, na), device=str(device)))
    return kept


# ── Main loop ─────────────────────────────────────────────────

def run_pg(model_name):
    model_cls, first_fc, fc_attrs, patience = MODEL_SPECS[model_name]
    ckpt = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt):
        print(f"⚠  {ckpt} not found — skipping"); return

    model = model_cls(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    base_macs, base_params = get_model_complexity_info(
        model, (3,32,32), as_strings=False, verbose=False)
    base_flops = 2 * base_macs

    fc_layer = PG_FC_LAYER[model_name]
    diffs    = load_pg_diffs(model_name, fc_layer)
    if diffs is None:
        print(f"⚠  No PG1 data for {model_name}/{fc_layer} — run pg_ext first"); return

    print(f"\n{'='*55}\nPG pruning  {model_name.upper()}  CIFAR-100\n{'='*55}")
    print(f"Device: {device}  batch={BATCH}  thresholds={THRESHOLDS}", flush=True)

    n_comp     = min(PCA_COMPS, diffs.shape[0], diffs.shape[1])
    pca        = PCA(n_components=n_comp).fit(diffs)
    components = pca.components_
    print(f"  PCA {n_comp} comps, explained var = {pca.explained_variance_ratio_.sum():.3f}",
          flush=True)

    print("  computing ASF-S/PG filter importances...", flush=True)
    fc_w        = getattr(model, fc_layer).weight.data.cpu().numpy()
    conv_seq    = SEQ[model_name]
    conv_attrs  = [t[0] for t in conv_seq]
    conv_flats  = [getattr(model, ca).weight.data.cpu().numpy()
                    .reshape(getattr(model, ca).out_channels, -1)
                   for ca in conv_attrs]
    importances = [filter_importance_gpu(cf, fc_w, components) for cf in conv_flats]

    csv_path    = os.path.join(RESULTS_DIR, f"pg_{model_name}_cifar100.csv")
    file_exists = os.path.exists(csv_path)
    out_dir     = os.path.join(RESULTS_DIR, "pruned_models", f"pg_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Conv2Net: 2-D threshold grid; others: single shared threshold
    if model_name == "conv2net":
        grid = list(itertools.product(THRESHOLDS, repeat=2))
        def run_one(args):
            thr1, thr2 = args
            m = copy.deepcopy(model).to(device)
            kept = {}
            for i, (ca, ba, na) in enumerate(conv_seq):
                thr = thr1 if i == 0 else thr2
                nc, idx = prune_conv(getattr(m,ca), importances[i]>thr,
                                     device=str(device))
                nb = prune_bn(getattr(m,ba), idx, device=str(device))
                setattr(m,ca,nc); setattr(m,ba,nb); kept[ca]=len(idx)
                if na: setattr(m,na,update_next_conv_in(idx,getattr(m,na),device=str(device)))
            return m, kept
        extra = lambda a: {"threshold_conv1": a[0], "threshold_conv2": a[1]}
        label = lambda a: f"({a[0]},{a[1]})"
        fname = lambda a: f"thr{a[0]:.2f}_{a[1]:.2f}.pth"
    else:
        grid  = [(t,) for t in THRESHOLDS]
        def run_one(args):
            m = copy.deepcopy(model).to(device)
            return m, apply_pruning(m, conv_seq, importances, args[0])
        extra = lambda a: {"threshold": a[0]}
        label = lambda a: str(a[0])
        fname = lambda a: f"thr{a[0]:.2f}.pth"

    print(f"  running {len(grid)} pruning setting(s)...", flush=True)

    for step, args in enumerate(grid, start=1):
        print(f"\n  [{step}/{len(grid)}] threshold={label(args)}: pruning layers...",
              flush=True)
        m, kept = run_one(args)
        m = rebuild_fc(m, first_fc, device=str(device))

        for ca, ba, _ in conv_seq:
            for p in getattr(m,ca).parameters(): p.requires_grad=False
            for p in getattr(m,ba).parameters(): p.requires_grad=False

        print(f"  [{step}/{len(grid)}] threshold={label(args)}: "
              f"kept={list(kept.values())}; evaluating before fine-tune...",
              flush=True)
        pre_acc  = eval_acc(m, val_loader, device=str(device))
        print(f"  [{step}/{len(grid)}] threshold={label(args)}: "
              f"pre_acc={pre_acc:.2f}; fine-tuning FC...",
              flush=True)
        best_acc, ep = fine_tune_fc(m, fc_attrs, train_loader, val_loader,
                                     epochs=200, lr=1e-4,
                                     patience=patience, device=str(device),
                                     verbose_every=1)

        print(f"  [{step}/{len(grid)}] threshold={label(args)}: "
              "measuring FLOPs/ECE and saving...",
              flush=True)
        macs, params = get_model_complexity_info(m, (3,32,32), as_strings=False, verbose=False)
        flops     = 2*macs
        ece       = compute_ece(m, val_loader, device=str(device))
        flops_red = 100*(base_flops-flops)/base_flops
        param_red = 100*(base_params-params)/base_params

        torch.save(m.state_dict(), os.path.join(out_dir, fname(args)))
        torch.cuda.empty_cache()

        row = dict(method="pg", model=model_name, seed=SEED,
                   **extra(args),
                   **{f"kept_{k}":v for k,v in kept.items()},
                   **{f"pruned_{k}_pct":100*(1-v/getattr(model,k).out_channels)
                      for k,v in kept.items()},
                   total_param_red_pct=param_red, pre_acc=pre_acc, val_acc=best_acc,
                   epochs=ep, early_stop=(ep<200),
                   flops=flops, flops_red_pct=flops_red, params=params, ece=ece)
        pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False,
                                    header=not file_exists, float_format="%.4f")
        file_exists = True
        print(f"  thr={label(args)}  kept={list(kept.values())}  "
              f"pre={pre_acc:.2f}  acc={best_acc:.2f}",
              flush=True)

    print(f"  Results → {csv_path}", flush=True)


# for mn in ["conv2net", "conv6net", "vgg16"]:
for mn in ["conv2net"]:
    run_pg(mn)

print("\n✅  PG pruning complete")
