#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Original-style PG pruning adapted to CIFAR-100 for:
  - Conv2Net
  - Conv6Net
  - LeNet300_100 (MLP)

Core idea matches your pg_pruning_conv2.py:
  1) Load PG diffs (correct - incorrect) from a chosen PG layer (default: logits layer for conv2/conv6, fc2 for lenet)
  2) PCA on PG diffs to get top components
  3) Compute importance per filter/neuron via projection against averaged downstream weights
  4) Physically prune the network (rebuild layers) and retrain/evaluate

Note:
- For Conv2Net/Conv6Net, pruning conv channels requires propagating kept indices into next layers.
- For LeNet, pruning neurons propagates into the next FC layer.
"""

import argparse, os, random, copy, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.decomposition import PCA

from models import Conv2Net, Conv6Net, LeNet300_100

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def cifar100_loaders(data_root, batch_size):
    mean=(0.5071, 0.4867, 0.4408)
    std =(0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_loader = DataLoader(datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(datasets.CIFAR100(root=data_root, train=False, download=True, transform=val_tf),
                            batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    tot, cor = 0,0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        cor += (pred==y).sum().item()
        tot += y.size(0)
    return 100.0*cor/max(tot,1)

def retrain(model, train_loader, val_loader, device, epochs=50, lr=0.01, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*epochs), int(0.8*epochs)], gamma=0.2)

    best = -1.0
    bad = 0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs+1):
        model.train()
        for x,y in tqdm(train_loader, desc=f"retrain ep {ep}/{epochs}", leave=False):
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = eval_acc(model, val_loader, device)
        if acc > best:
            best = acc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return best, ep

def load_pg_diffs(pg_folder, num_classes, pg_layer_name):
    diffs = []
    for cls in range(num_classes):
        pc = os.path.join(pg_folder, "correct", "pg1_data", f"{cls}_{pg_layer_name}_pg1.npy")
        pi = os.path.join(pg_folder, "incorrect", "pg1_data", f"{cls}_{pg_layer_name}_pg1.npy")
        if not (os.path.exists(pc) and os.path.exists(pi)):
            continue
        diffs.append(np.load(pc) - np.load(pi))
    diffs = np.array(diffs)
    if diffs.size == 0:
        raise RuntimeError(f"No PG diff data found in {pg_folder} for layer {pg_layer_name}")
    return diffs

def compute_importance(conv_flat, downstream_w, components):
    """
    Matches your original compute_importance:
      avg_down = mean over classes of downstream weights
      fc_proj = comp * avg_down (truncate)
      importance(filter) = |mean_k (conv_flat @ fc_proj)_k|
    """
    avg_down = np.mean(downstream_w, axis=0)
    scores = []
    for comp in components:
        proj_size = min(conv_flat.shape[1], avg_down.shape[0], comp.shape[0])
        conv_proj = conv_flat[:, :proj_size]
        vec = comp[:proj_size] * avg_down[:proj_size]
        scores.append(conv_proj @ vec)
    scores = np.array(scores)
    imp = np.abs(scores.mean(axis=0))
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
    return imp

def prune_conv(conv: nn.Conv2d, keep_mask, keep_in_idx=None, device="cpu"):
    W = conv.weight.data.detach().cpu().numpy()
    keep_mask = np.array(keep_mask, dtype=bool)
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    keep_idx = np.where(keep_mask)[0]

    if keep_in_idx is not None:
        W_new = W[keep_idx][:, keep_in_idx]
        in_ch = len(keep_in_idx)
    else:
        W_new = W[keep_idx]
        in_ch = W_new.shape[1]

    b_new = conv.bias.data.detach().cpu().numpy()[keep_idx] if conv.bias is not None else None

    new_conv = nn.Conv2d(in_ch, len(keep_idx), conv.kernel_size, stride=conv.stride, padding=conv.padding,
                         bias=(conv.bias is not None)).to(device)
    new_conv.weight.data = torch.tensor(W_new, device=device, dtype=conv.weight.dtype)
    if b_new is not None:
        new_conv.bias.data = torch.tensor(b_new, device=device, dtype=conv.bias.dtype)
    return new_conv, keep_idx

def prune_bn(bn: nn.BatchNorm2d, keep_idx, device="cpu"):
    new_bn = nn.BatchNorm2d(len(keep_idx)).to(device)
    new_bn.weight.data = bn.weight.data[keep_idx].clone().to(device)
    new_bn.bias.data = bn.bias.data[keep_idx].clone().to(device)
    new_bn.running_mean = bn.running_mean[keep_idx].clone().to(device)
    new_bn.running_var  = bn.running_var[keep_idx].clone().to(device)
    return new_bn

def prune_linear(fc: nn.Linear, keep_mask, keep_in_idx=None, device="cpu"):
    W = fc.weight.data.detach().cpu().numpy()
    keep_mask = np.array(keep_mask, dtype=bool)
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    keep_idx = np.where(keep_mask)[0]
    if keep_in_idx is not None:
        W_new = W[keep_idx][:, keep_in_idx]
        in_f = len(keep_in_idx)
    else:
        W_new = W[keep_idx]
        in_f = W_new.shape[1]
    b_new = fc.bias.data.detach().cpu().numpy()[keep_idx] if fc.bias is not None else None

    new_fc = nn.Linear(in_f, len(keep_idx), bias=(fc.bias is not None)).to(device)
    new_fc.weight.data = torch.tensor(W_new, device=device, dtype=fc.weight.dtype)
    if b_new is not None:
        new_fc.bias.data = torch.tensor(b_new, device=device, dtype=fc.bias.dtype)
    return new_fc, keep_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pg_root", required=True, help="Root that contains <model>/correct/pg1_data etc.")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs_retrain", type=int, default=80)
    ap.add_argument("--lr_retrain", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--pca_n_components", type=int, default=4)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    ap.add_argument("--out_dir", default="./pruned_checkpoints_cifar100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pg_layer", default="", help="Which PG layer to use. Default: conv2net=fc, conv6net=fc2, lenet=fc2")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # loaders
    train_loader, val_loader = cifar100_loaders(args.data_root, args.batch_size)

    # load model
    if args.model == "conv2net":
        model = Conv2Net(num_classes=100).to(device)
        pg_layer = args.pg_layer or "fc"
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck, strict=True)
        # weights for importance
        conv1_flat = model.conv1.weight.data.detach().cpu().numpy().reshape(model.conv1.out_channels, -1)
        conv2_flat = model.conv2.weight.data.detach().cpu().numpy().reshape(model.conv2.out_channels, -1)
        fc_w = model.fc.weight.data.detach().cpu().numpy()

        pg_folder = os.path.join(args.pg_root, "conv2net")
        pgdiff = load_pg_diffs(pg_folder, 100, pg_layer)
        pca = PCA(n_components=min(args.pca_n_components, pgdiff.shape[1]))
        pca.fit(pgdiff)
        comps = pca.components_

        imp1 = compute_importance(conv1_flat, fc_w, comps)
        imp2 = compute_importance(conv2_flat, fc_w, comps)

        for th1 in args.thresholds:
            for th2 in args.thresholds:
                m = copy.deepcopy(model)

                keep1 = (imp1 >= th1)
                keep2 = (imp2 >= th2)

                # prune conv1 -> conv2 input
                m.conv1, idx1 = prune_conv(m.conv1, keep1, keep_in_idx=None, device=device)
                m.bn1 = prune_bn(m.bn1, idx1, device=device)
                # prune conv2 (inputs follow idx1)
                m.conv2, idx2 = prune_conv(m.conv2, keep2, keep_in_idx=idx1, device=device)
                m.bn2 = prune_bn(m.bn2, idx2, device=device)

                # adjust FC input (after pool: 16x16)
                in_features = len(idx2) * 16 * 16
                old_fc = m.fc
                W = old_fc.weight.data.detach().cpu().numpy()
                # select only kept conv2 channels' spatial slices
                W_new = W.reshape(W.shape[0], -1, 16, 16)[:, idx2].reshape(W.shape[0], -1)
                m.fc = nn.Linear(in_features, 100, bias=(old_fc.bias is not None)).to(device)
                m.fc.weight.data = torch.tensor(W_new, device=device, dtype=old_fc.weight.dtype)
                if old_fc.bias is not None:
                    m.fc.bias.data = old_fc.bias.data.clone().to(device)

                pre = eval_acc(m, val_loader, device)
                best, stop_ep = retrain(m, train_loader, val_loader, device, epochs=args.epochs_retrain, lr=args.lr_retrain, patience=args.patience)
                print(f"[conv2net] th1={th1} th2={th2} pre={pre:.2f}% post={best:.2f}% stop_ep={stop_ep}")

                outp = os.path.join(args.out_dir, f"cifar100_conv2net_th1{th1}_th2{th2}.pth")
                torch.save({"model_state_dict": m.state_dict(), "thresholds": [th1, th2], "post_acc": best}, outp)

    elif args.model == "conv6net":
        model = Conv6Net(num_classes=100).to(device)
        pg_layer = args.pg_layer or "fc2"
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck, strict=True)

        # downstream weights: use fc2 as in original spirit (logits layer)
        fc2_w = model.fc2.weight.data.detach().cpu().numpy()

        # choose conv layers to prune (full original-style pruning can be huge; here we prune three convs and propagate)
        # You can extend this list if you want.
        convs = [("conv2","bn2","conv3"), ("conv4","bn4","conv5"), ("conv6","bn6","fc1")]
        flats = {
            "conv2": model.conv2.weight.data.detach().cpu().numpy().reshape(model.conv2.out_channels, -1),
            "conv4": model.conv4.weight.data.detach().cpu().numpy().reshape(model.conv4.out_channels, -1),
            "conv6": model.conv6.weight.data.detach().cpu().numpy().reshape(model.conv6.out_channels, -1),
        }

        pg_folder = os.path.join(args.pg_root, "conv6net")
        pgdiff = load_pg_diffs(pg_folder, 100, pg_layer)
        pca = PCA(n_components=min(args.pca_n_components, pgdiff.shape[1]))
        pca.fit(pgdiff)
        comps = pca.components_

        imp = {k: compute_importance(v, fc2_w, comps) for k,v in flats.items()}

        for th in args.thresholds:
            m = copy.deepcopy(model)

            # Start from input channels of each segment.
            # Segment 1: conv2 output pruned -> affects bn2 and conv3 input
            keep2 = (imp["conv2"] >= th)
            m.conv2, idx2 = prune_conv(m.conv2, keep2, keep_in_idx=None, device=device)  # conv2 input is conv1 out (32) unchanged
            m.bn2 = prune_bn(m.bn2, idx2, device=device)
            # conv3 input channels must match idx2
            m.conv3, idx3in = prune_conv(m.conv3, keep_mask=np.ones(m.conv3.out_channels, dtype=bool), keep_in_idx=idx2, device=device)
            m.bn3 = prune_bn(m.bn3, idx3in, device=device)  # idx3in are output idx since keep all outputs; same length as out_channels

            # Segment 2: conv4 output pruned -> affects bn4 and conv5 input
            keep4 = (imp["conv4"] >= th)
            # conv4 input is conv3 output channels (64) unchanged (we didn't prune conv3 output), so keep_in_idx=None
            m.conv4, idx4 = prune_conv(m.conv4, keep4, keep_in_idx=None, device=device)
            m.bn4 = prune_bn(m.bn4, idx4, device=device)
            # conv5 input channels match idx4
            m.conv5, idx5in = prune_conv(m.conv5, keep_mask=np.ones(m.conv5.out_channels, dtype=bool), keep_in_idx=idx4, device=device)
            m.bn5 = prune_bn(m.bn5, idx5in, device=device)

            # Segment 3: conv6 output pruned -> affects bn6 and fc1 input
            keep6 = (imp["conv6"] >= th)
            m.conv6, idx6 = prune_conv(m.conv6, keep6, keep_in_idx=None, device=device)  # input from conv5 output (128) unchanged
            m.bn6 = prune_bn(m.bn6, idx6, device=device)

            # adjust fc1 input features (after pool3: 4x4)
            old_fc1 = m.fc1
            W1 = old_fc1.weight.data.detach().cpu().numpy()  # (256, 128*4*4)
            W1 = W1.reshape(W1.shape[0], -1, 4, 4)[:, idx6].reshape(W1.shape[0], -1)
            m.fc1 = nn.Linear(len(idx6)*4*4, old_fc1.out_features, bias=(old_fc1.bias is not None)).to(device)
            m.fc1.weight.data = torch.tensor(W1, device=device, dtype=old_fc1.weight.dtype)
            if old_fc1.bias is not None:
                m.fc1.bias.data = old_fc1.bias.data.clone().to(device)

            pre = eval_acc(m, val_loader, device)
            best, stop_ep = retrain(m, train_loader, val_loader, device, epochs=args.epochs_retrain, lr=args.lr_retrain, patience=args.patience)
            print(f"[conv6net] th={th} pre={pre:.2f}% post={best:.2f}% stop_ep={stop_ep}")

            outp = os.path.join(args.out_dir, f"cifar100_conv6net_th{th}.pth")
            torch.save({"model_state_dict": m.state_dict(), "threshold": th, "post_acc": best}, outp)

    else:
        model = LeNet300_100(input_size=3*32*32, num_classes=100).to(device)
        pg_layer = args.pg_layer or "fc2"
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck, strict=True)

        # downstream: fc3 weights (logits)
        fc3_w = model.fc3.weight.data.detach().cpu().numpy()

        fc1_flat = model.fc1.weight.data.detach().cpu().numpy()  # (300, 3072)
        fc2_flat = model.fc2.weight.data.detach().cpu().numpy()  # (100, 300)

        pg_folder = os.path.join(args.pg_root, "lenet")
        pgdiff = load_pg_diffs(pg_folder, 100, pg_layer)
        pca = PCA(n_components=min(args.pca_n_components, pgdiff.shape[1]))
        pca.fit(pgdiff)
        comps = pca.components_

        imp1 = compute_importance(fc1_flat, fc3_w, comps)
        imp2 = compute_importance(fc2_flat, fc3_w, comps)

        for th1 in args.thresholds:
            for th2 in args.thresholds:
                m = copy.deepcopy(model)
                keep1 = (imp1 >= th1)
                keep2 = (imp2 >= th2)

                # prune fc1 -> fc2 input
                m.fc1, idx1 = prune_linear(m.fc1, keep1, keep_in_idx=None, device=device)
                # prune fc2 (inputs follow idx1)
                m.fc2, idx2 = prune_linear(m.fc2, keep2, keep_in_idx=idx1, device=device)
                # adjust fc3 input
                old_fc3 = m.fc3
                W3 = old_fc3.weight.data.detach().cpu().numpy()[:, idx2]
                m.fc3 = nn.Linear(len(idx2), 100, bias=(old_fc3.bias is not None)).to(device)
                m.fc3.weight.data = torch.tensor(W3, device=device, dtype=old_fc3.weight.dtype)
                if old_fc3.bias is not None:
                    m.fc3.bias.data = old_fc3.bias.data.clone().to(device)

                pre = eval_acc(m, val_loader, device)
                best, stop_ep = retrain(m, train_loader, val_loader, device, epochs=args.epochs_retrain, lr=args.lr_retrain, patience=args.patience)
                print(f"[lenet] th1={th1} th2={th2} pre={pre:.2f}% post={best:.2f}% stop_ep={stop_ep}")

                outp = os.path.join(args.out_dir, f"cifar100_lenet_th1{th1}_th2{th2}.pth")
                torch.save({"model_state_dict": m.state_dict(), "thresholds": [th1, th2], "post_acc": best}, outp)

if __name__ == "__main__":
    main()
