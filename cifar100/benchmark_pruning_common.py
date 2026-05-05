"""
Shared CIFAR-100 structured pruning benchmark runner.

The method scripts provide a per-convolution importance score. This runner
handles exact per-layer filter pruning, FC rebuilding, FC fine-tuning,
metric logging, and checkpoint saving for Conv2Net, Conv6Net, and VGG-16.
"""

import argparse
import copy
import os
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (CIFAR100_MEAN, CIFAR100_STD, DATA_DIR, NUM_CLASSES,
                    RESULTS_DIR, TRAINED_MODELS_DIR)
from models import Conv2Net, Conv6Net, VGG16CIFAR
from pruning_utils import (compute_ece, eval_acc, fine_tune_fc, prune_bn,
                           prune_conv, rebuild_fc, update_next_conv_in)

SEED = 42
BATCH = 128
DEFAULT_RATIOS = [0.7]
DEFAULT_MODELS = ["conv2net", "conv6net", "vgg16"]

VGG_SEQ = [
    ("conv1_1", "bn1_1", "conv1_2"), ("conv1_2", "bn1_2", "conv2_1"),
    ("conv2_1", "bn2_1", "conv2_2"), ("conv2_2", "bn2_2", "conv3_1"),
    ("conv3_1", "bn3_1", "conv3_2"), ("conv3_2", "bn3_2", "conv3_3"),
    ("conv3_3", "bn3_3", "conv4_1"), ("conv4_1", "bn4_1", "conv4_2"),
    ("conv4_2", "bn4_2", "conv4_3"), ("conv4_3", "bn4_3", "conv5_1"),
    ("conv5_1", "bn5_1", "conv5_2"), ("conv5_2", "bn5_2", "conv5_3"),
    ("conv5_3", "bn5_3", None),
]
CONV6_SEQ = [
    ("conv1", "bn1", "conv2"), ("conv2", "bn2", "conv3"),
    ("conv3", "bn3", "conv4"), ("conv4", "bn4", "conv5"),
    ("conv5", "bn5", "conv6"), ("conv6", "bn6", None),
]
CONV2_SEQ = [("conv1", "bn1", "conv2"), ("conv2", "bn2", None)]

SEQ = {"conv2net": CONV2_SEQ, "conv6net": CONV6_SEQ, "vgg16": VGG_SEQ}

MODEL_SPECS = {
    "conv2net": (Conv2Net, "fc", ["fc"], 10),
    "conv6net": (Conv6Net, "fc1", ["fc1", "fc2"], 10),
    "vgg16": (VGG16CIFAR, "fc1", ["fc1", "fc2"], 15),
}

ScoreFn = Callable[[torch.nn.Module, str, list, DataLoader, torch.device, int],
                   dict[str, np.ndarray]]


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(device, batch_size=BATCH):
    pin_memory = device.type == "cuda"
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_loader = DataLoader(
        datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=train_tf),
        batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True)
    val_loader = DataLoader(
        datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf),
        batch_size=256, shuffle=False, num_workers=4,
        pin_memory=pin_memory, persistent_workers=True)
    return train_loader, val_loader


def parse_args(method_name):
    parser = argparse.ArgumentParser(
        description=f"{method_name.upper()} CIFAR-100 structured pruning benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS,
                        help="Model architectures to run.")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS,
                        help="Per-layer filter pruning ratios. Default: 0.7")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max FC fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH,
                        help="Training batch size.")
    parser.add_argument("--score-batches", type=int, default=5,
                        help="Training batches used by data-dependent scorers.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="FC fine-tuning learning rate.")
    parser.add_argument("--verbose-every", type=int, default=1,
                        help="Print every N fine-tuning epochs; 0 disables.")
    return parser.parse_args()


def load_base_model(model_name, device):
    model_cls, _, _, _ = MODEL_SPECS[model_name]
    ckpt = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt):
        print(f"WARNING: {ckpt} not found; skipping {model_name}", flush=True)
        return None
    model = model_cls(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def keep_mask_from_scores(scores, prune_ratio):
    scores = np.asarray(scores, dtype=np.float64)
    out_channels = scores.shape[0]
    keep_count = max(1, int(round(out_channels * (1.0 - prune_ratio))))
    keep_count = min(out_channels, keep_count)
    mask = np.zeros(out_channels, dtype=bool)
    mask[np.argsort(-scores)[:keep_count]] = True
    return mask


def apply_structured_pruning(model, base_model, conv_seq, scores_by_layer,
                             prune_ratio, device):
    kept = {}
    for conv_a, bn_a, next_a in conv_seq:
        conv = getattr(model, conv_a)
        scores = scores_by_layer.get(conv_a)
        if scores is None:
            scores = np.ones(getattr(base_model, conv_a).out_channels)
        mask = keep_mask_from_scores(scores, prune_ratio)

        nc, idx = prune_conv(conv, mask, device=str(device))
        nb = prune_bn(getattr(model, bn_a), idx, device=str(device))
        setattr(model, conv_a, nc)
        setattr(model, bn_a, nb)
        kept[conv_a] = len(idx)

        if next_a:
            next_conv = update_next_conv_in(idx, getattr(model, next_a),
                                            device=str(device))
            setattr(model, next_a, next_conv)
    return kept


def run_method(method_name, score_fn: ScoreFn, args=None):
    if args is None:
        args = parse_args(method_name)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(device, batch_size=args.batch_size)

    print(f"Device: {device} | method={method_name} | ratios={args.ratios}",
          flush=True)

    for model_name in args.models:
        model_cls, first_fc, fc_attrs, patience = MODEL_SPECS[model_name]
        base = load_base_model(model_name, device)
        if base is None:
            continue

        base_macs, base_params = get_model_complexity_info(
            base, (3, 32, 32), as_strings=False, verbose=False)
        base_flops = 2 * base_macs
        conv_seq = SEQ[model_name]

        print(f"\n{'=' * 55}\n{method_name.upper()} {model_name.upper()} CIFAR-100\n{'=' * 55}",
              flush=True)
        print("Computing filter scores...", flush=True)
        scores_by_layer = score_fn(base, model_name, conv_seq, train_loader,
                                   device, args.score_batches)

        csv_path = os.path.join(RESULTS_DIR,
                                f"{method_name}_{model_name}_cifar100.csv")
        file_exists = os.path.exists(csv_path)
        out_dir = os.path.join(RESULTS_DIR, "pruned_models",
                               f"{method_name}_{model_name}")
        os.makedirs(out_dir, exist_ok=True)

        for ratio in args.ratios:
            print(f"\nratio={ratio:.2f}: pruning layers...", flush=True)
            m = copy.deepcopy(base).to(device)
            kept = apply_structured_pruning(m, base, conv_seq, scores_by_layer,
                                            ratio, device)
            m = rebuild_fc(m, first_fc, device=str(device))

            print(f"ratio={ratio:.2f}: kept={list(kept.values())}; evaluating...",
                  flush=True)
            pre_acc = eval_acc(m, val_loader, device=str(device))
            print(f"ratio={ratio:.2f}: pre_acc={pre_acc:.2f}; fine-tuning FC...",
                  flush=True)
            best_acc, ep = fine_tune_fc(
                m, fc_attrs, train_loader, val_loader, epochs=args.epochs,
                lr=args.lr, patience=patience, device=str(device),
                verbose_every=args.verbose_every)

            print(f"ratio={ratio:.2f}: measuring FLOPs/ECE and saving...",
                  flush=True)
            macs, params = get_model_complexity_info(
                m, (3, 32, 32), as_strings=False, verbose=False)
            flops = 2 * macs
            ece = compute_ece(m, val_loader, device=str(device))
            flops_red = 100 * (base_flops - flops) / base_flops
            param_red = 100 * (base_params - params) / base_params

            torch.save(m.state_dict(), os.path.join(out_dir,
                       f"ratio{ratio:.2f}.pth"))
            if device.type == "cuda":
                torch.cuda.empty_cache()

            row = dict(
                method=method_name, model=model_name, prune_ratio=ratio,
                seed=SEED, pre_acc=pre_acc, val_acc=best_acc, epochs=ep,
                early_stop=(ep < args.epochs), flops=flops,
                flops_red_pct=flops_red, params=params,
                param_red_pct=param_red, ece=ece,
                **{f"kept_{k}": v for k, v in kept.items()},
                **{f"pruned_{k}_pct":
                   100 * (1 - v / getattr(base, k).out_channels)
                   for k, v in kept.items()})
            pd.DataFrame([row]).to_csv(
                csv_path, mode="a", index=False, header=not file_exists,
                float_format="%.4f")
            file_exists = True
            print(f"ratio={ratio:.2f} pre={pre_acc:.2f} acc={best_acc:.2f} "
                  f"flops_down={flops_red:.2f}% params_down={param_red:.2f}%",
                  flush=True)

        print(f"Results -> {csv_path}", flush=True)
