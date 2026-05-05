#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PG-based Conv2Net pruning benchmarking script (dual conv threshold grid)
- Tests all combos of thresholds for conv1 & conv2
- Retrains the whole model 
- Logs pre- and post-retrain accuracy, FLOPS, params, ECE, pruning %, and early stop epoch
- Appends to CSV as it goes for crash-proof logging
"""

import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from ptflops import get_model_complexity_info
from models import Conv2Net
from sklearn.decomposition import PCA
import os
import random

# ----------------------------
# Seed for reproducibility
# ----------------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# Settings
# ----------------------------
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_classes = 10
pg_folder = Path("/home/mindula/Desktop/moving_forward_results/pg_multi_models/conv2net/")
model_path = "/home/mindula/Desktop/moving_forward_results/trained_models_new/conv2_best.pth"

prune_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
patience = 10
epochs_max = 200
batch_size = 128
pca_n_components = 4

csv_path = "/home/mindula/Desktop/moving_forward_results/pg_pruning_results_conv2.csv"
file_exists = os.path.exists(csv_path)

# ----------------------------
# Load model
# ----------------------------
model = Conv2Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

conv1_w = model.conv1.weight.data.cpu().numpy()
conv2_w = model.conv2.weight.data.cpu().numpy()
fc_w = model.fc.weight.data.cpu().numpy()

conv1_flat = conv1_w.reshape(conv1_w.shape[0], -1)
conv2_flat = conv2_w.reshape(conv2_w.shape[0], -1)

# ----------------------------
# Load PG differences
# ----------------------------
fc_pg_diff_all = []
for cls in range(num_classes):
    pg_c = pg_folder / f"correct/pg1_data/{cls}_fc_pg1.npy"
    pg_i = pg_folder / f"incorrect/pg1_data/{cls}_fc_pg1.npy"
    if not pg_c.exists() or not pg_i.exists():
        continue
    pg_c = np.load(pg_c)
    pg_i = np.load(pg_i)
    fc_pg_diff_all.append(pg_c - pg_i)

fc_pg_diff_all = np.array(fc_pg_diff_all)
if fc_pg_diff_all.size == 0:
    raise RuntimeError("No PG diff data found.")

# ----------------------------
# PCA projection
# ----------------------------
pca = PCA(n_components=min(pca_n_components, fc_pg_diff_all.shape[1]))
pca.fit(fc_pg_diff_all)
top_components = pca.components_

def compute_importance(conv_flat, fc_w, components):
    avg_fc = np.mean(fc_w, axis=0)
    scores = []
    for comp in components:
        proj_size = min(conv_flat.shape[1], avg_fc.shape[0], comp.shape[0])
        conv_proj = conv_flat[:, :proj_size]
        fc_proj = comp[:proj_size] * avg_fc[:proj_size]
        contrib = conv_proj @ fc_proj
        scores.append(contrib)
    scores = np.array(scores)
    importance = np.abs(scores.mean(axis=0))
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-12)
    return importance

importance_conv1 = compute_importance(conv1_flat, fc_w, top_components)
importance_conv2 = compute_importance(conv2_flat, fc_w, top_components)

# ----------------------------
# CIFAR-10 loaders
# ----------------------------
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])
train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=val_tf),
                        batch_size=batch_size, shuffle=False)

# ----------------------------
# Helper functions
# ----------------------------
def prune_conv(conv, keep_mask, keep_in_idx=None):
    W = conv.weight.data.cpu().numpy()
    keep_mask = np.array(keep_mask, dtype=bool)
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    keep_idx = np.where(keep_mask)[0]
    if keep_in_idx is not None:
        W_new = W[keep_idx][:, keep_in_idx]
    else:
        W_new = W[keep_idx]
    bias_new = conv.bias.data.cpu().numpy()[keep_idx] if conv.bias is not None else None

    new_conv = nn.Conv2d(W_new.shape[1], W_new.shape[0], conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         bias=(conv.bias is not None)).to(device)
    new_conv.weight.data = torch.tensor(W_new, device=device, dtype=conv.weight.dtype)
    if bias_new is not None:
        new_conv.bias.data = torch.tensor(bias_new, device=device, dtype=conv.bias.dtype)
    return new_conv, keep_idx

def prune_bn(bn, keep_idx):
    new_bn = nn.BatchNorm2d(len(keep_idx)).to(device)
    new_bn.weight.data = bn.weight.data[keep_idx].clone().to(device)
    new_bn.bias.data = bn.bias.data[keep_idx].clone().to(device)
    new_bn.running_mean = bn.running_mean[keep_idx].clone().to(device)
    new_bn.running_var = bn.running_var[keep_idx].clone().to(device)
    return new_bn

def adjust_fc(model, conv1, bn1, conv2, bn2):
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32).to(device)
        x = F.relu(bn1(conv1(dummy)))
        x = F.relu(bn2(conv2(x)))
        x = model.pool(x)
        new_in_features = x.numel()
    old_fc = model.fc
    new_fc = nn.Linear(new_in_features, old_fc.out_features).to(device)
    min_in = min(old_fc.in_features, new_in_features)
    new_fc.weight.data[:, :min_in] = old_fc.weight.data[:, :min_in]
    new_fc.bias.data = old_fc.bias.data.clone()
    return new_fc

def eval_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return 100. * correct / total

def fine_tune(model, train_loader, val_loader, epochs, patience):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val, no_improve = 0, 0
    best_state = None
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = eval_accuracy(model, val_loader)
        print(f"[Epoch {ep+1}] Val Acc: {val_acc:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1}.")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_val, ep + 1  # return best acc and epochs used

def compute_ece(model, loader, n_bins=15):
    model.eval()
    confs, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)
            conf, pred = probs.max(1)
            confs.append(conf)
            preds.append(pred)
            labels.append(y)
    confs = torch.cat(confs)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    ece = 0.0
    for low in torch.linspace(0, 1, n_bins + 1)[:-1]:
        high = low + 1 / n_bins
        mask = (confs > low) & (confs <= high)
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).float().mean()
            avg_conf = confs[mask].mean()
            ece += (mask.sum().float() / labels.size(0)) * torch.abs(acc - avg_conf)
    return ece.item()

# ----------------------------
# Baseline params
# ----------------------------
base_macs, base_params = get_model_complexity_info(model, (3,32,32), as_strings=False, verbose=False)

# ----------------------------
# Run pruning grid
# ----------------------------
for thr1, thr2 in itertools.product(prune_thresholds, prune_thresholds):
    print(f"\n=== conv1<{thr1:.2f}, conv2<{thr2:.2f} ===")
    keep1 = importance_conv1 > thr1
    keep2 = importance_conv2 > thr2

    m = copy.deepcopy(model).to(device)
    pconv1, idx1 = prune_conv(m.conv1, keep1)
    pbn1 = prune_bn(m.bn1, idx1)
    m.conv1, m.bn1 = pconv1, pbn1

    pconv2, idx2 = prune_conv(m.conv2, keep2, keep_in_idx=idx1)
    pbn2 = prune_bn(m.bn2, idx2)
    m.conv2, m.bn2 = pconv2, pbn2

    m.fc = adjust_fc(m, pconv1, pbn1, pconv2, pbn2)
    m.to(device)
    for p in m.parameters():
        p.requires_grad = True

    # --- Pre-retrain accuracy ---
    pre_acc = eval_accuracy(m, val_loader)
    print(f"Pre-retrain Accuracy: {pre_acc:.2f}%")

    # --- Fine-tune ---
    best_acc, epochs_used = fine_tune(m, train_loader, val_loader, epochs_max, patience)
    macs, params = get_model_complexity_info(m, (3,32,32), as_strings=False, verbose=False)
    flops = 2 * macs
    ece = compute_ece(m, val_loader)

    # --- Pruning stats ---
    total_conv1 = len(importance_conv1)
    total_conv2 = len(importance_conv2)
    kept1 = int(keep1.sum())
    kept2 = int(keep2.sum())
    pruned1_pct = 100.0 * (1 - kept1 / total_conv1)
    pruned2_pct = 100.0 * (1 - kept2 / total_conv2)
    total_pruned_pct = 100.0 * (1 - params / base_params)

    # --- Append result to CSV immediately ---
    result = {
        "conv1_thr": thr1,
        "conv2_thr": thr2,
        "kept_conv1": kept1,
        "kept_conv2": kept2,
        "pruned_conv1_%": pruned1_pct,
        "pruned_conv2_%": pruned2_pct,
        "total_pruned_%": total_pruned_pct,
        "pre_acc": pre_acc,
        "best_acc": best_acc,
        "epochs_till_stop": epochs_used,
        "flops": flops,
        "params": params,
        "ece": ece
    }
    df = pd.DataFrame([result])
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
    file_exists = True

print("\n💾 Results saved to pg_pruning_results_conv2.csv ✅")

