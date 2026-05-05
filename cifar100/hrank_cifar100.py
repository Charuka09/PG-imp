"""
hrank_cifar100.py – HRank structured pruning on GPU for all three CIFAR-100 models.

All heavy ops run on GPU:
  • HRank feature-map SVD     (pruning_utils.hrank_scores)
  • FC fine-tuning with AMP   (pruning_utils.fine_tune_fc)
  • DataLoaders with pin_memory + non_blocking transfers
"""

import copy, os, random
import numpy as np
import pandas as pd
import torch
from ptflops import get_model_complexity_info
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (TRAINED_MODELS_DIR, RESULTS_DIR, DATA_DIR,
                    CIFAR100_MEAN, CIFAR100_STD, NUM_CLASSES)
from models import Conv2Net, Conv6Net, VGG16CIFAR
from pruning_utils import (prune_conv, prune_bn, update_next_conv_in,
                            rebuild_fc, eval_acc, compute_ece,
                            fine_tune_fc, hrank_scores)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRUNE_RATIOS = [0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
BATCH        = 128
PIN_MEMORY   = (device.type == "cuda")

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomRotation(15),
    transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
val_tf = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
train_loader = DataLoader(
    datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf),
    batch_size=BATCH, shuffle=True,  num_workers=4,
    pin_memory=PIN_MEMORY, persistent_workers=True)
val_loader = DataLoader(
    datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf),
    batch_size=256,   shuffle=False, num_workers=4,
    pin_memory=PIN_MEMORY, persistent_workers=True)

# ── Conv layer sequences ──────────────────────────────────────
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
    "conv2net": (Conv2Net,    "fc",  ["fc"],          10),
    "conv6net": (Conv6Net,    "fc1", ["fc1","fc2"],   10),
    "vgg16":    (VGG16CIFAR,  "fc1", ["fc1","fc2"],   15),
}


def run_hrank(model_name):
    model_cls, first_fc, fc_attrs, patience = MODEL_SPECS[model_name]
    ckpt = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt):
        print(f"⚠  {ckpt} not found — skipping"); return

    base = model_cls(NUM_CLASSES).to(device)
    base.load_state_dict(torch.load(ckpt, map_location=device))
    base.eval()

    base_macs, base_params = get_model_complexity_info(
        base, (3,32,32), as_strings=False, verbose=False)
    base_flops = 2 * base_macs

    csv_path    = os.path.join(RESULTS_DIR, f"hrank_{model_name}_cifar100.csv")
    file_exists = os.path.exists(csv_path)
    out_dir     = os.path.join(RESULTS_DIR, "pruned_models", f"hrank_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*55}\nHRank  {model_name.upper()}  CIFAR-100\n{'='*55}")
    print(f"Device: {device}  batch={BATCH}  ratios={PRUNE_RATIOS}", flush=True)

    for ratio in PRUNE_RATIOS:
        print(f"\n  ratio={ratio:.2f}: computing HRank scores...", flush=True)
        m      = copy.deepcopy(base).to(device)
        scores = hrank_scores(m, train_loader, device=device, num_batches=5)
        kept   = {}

        print(f"  ratio={ratio:.2f}: pruning layers...", flush=True)
        for conv_a, bn_a, next_a in SEQ[model_name]:
            conv = getattr(m, conv_a)
            k    = max(1, int(conv.out_channels * (1 - ratio)))
            imp  = scores.get(conv_a, np.ones(conv.out_channels))
            mask = np.zeros(conv.out_channels, dtype=bool)
            mask[np.argsort(-imp)[:k]] = True

            nc, idx = prune_conv(conv, mask, device=str(device))
            nb      = prune_bn(getattr(m, bn_a), idx, device=str(device))
            setattr(m, conv_a, nc); setattr(m, bn_a, nb)
            kept[conv_a] = len(idx)

            if next_a:
                setattr(m, next_a,
                        update_next_conv_in(idx, getattr(m, next_a), device=str(device)))

        m = rebuild_fc(m, first_fc, device=str(device))

        print(f"  ratio={ratio:.2f}: evaluating before fine-tune...", flush=True)
        pre_acc  = eval_acc(m, val_loader, device=str(device))
        print(f"  ratio={ratio:.2f}: pre_acc={pre_acc:.2f}; fine-tuning FC...", flush=True)
        best_acc, ep = fine_tune_fc(m, fc_attrs, train_loader, val_loader,
                                     epochs=200, lr=1e-4,
                                     patience=patience, device=str(device),
                                     verbose_every=1)

        print(f"  ratio={ratio:.2f}: measuring FLOPs/ECE and saving...", flush=True)
        macs, params = get_model_complexity_info(m, (3,32,32), as_strings=False, verbose=False)
        flops     = 2 * macs
        ece       = compute_ece(m, val_loader, device=str(device))
        flops_red = 100*(base_flops-flops)/base_flops
        param_red = 100*(base_params-params)/base_params

        torch.save(m.state_dict(), os.path.join(out_dir, f"ratio{ratio:.2f}.pth"))
        torch.cuda.empty_cache()

        row = dict(method="hrank", model=model_name, prune_ratio=ratio, seed=SEED,
                   pre_acc=pre_acc, val_acc=best_acc, epochs=ep, early_stop=(ep<200),
                   flops=flops, flops_red_pct=flops_red,
                   params=params, param_red_pct=param_red, ece=ece,
                   **{f"kept_{k}": v for k,v in kept.items()},
                   **{f"pruned_{k}_pct": 100*(1-v/getattr(base,k).out_channels)
                      for k,v in kept.items()})
        pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False,
                                    header=not file_exists, float_format="%.4f")
        file_exists = True
        print(f"  ratio={ratio:.2f}  pre={pre_acc:.2f}  acc={best_acc:.2f}  "
              f"flops↓{flops_red:.1f}%  params↓{param_red:.1f}%")

    print(f"  Results → {csv_path}")


# for mn in ["conv2net", "conv6net", "vgg16"]:
for mn in ["conv2net"]:
    run_hrank(mn)

print("\nHRank pruning complete")
