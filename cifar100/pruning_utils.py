"""
pruning_utils.py – Shared helpers for all pruning scripts.
All heavy compute (SVD, fine-tuning) runs on GPU.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast


# ── Structural pruning ────────────────────────────────────────

def prune_conv(conv, keep_mask, keep_in_idx=None, device="cpu"):
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    idx = np.where(keep_mask)[0]
    W   = conv.weight.data.cpu().numpy()[idx]
    if keep_in_idx is not None:
        # Filter keep_in_idx to only include valid indices for the current input channels
        keep_in_idx = np.array([i for i in keep_in_idx if i < W.shape[1]])
        if len(keep_in_idx) > 0:
            W = W[:, keep_in_idx]
    bias = conv.bias.data.cpu().numpy()[idx] if conv.bias is not None else None
    new  = nn.Conv2d(W.shape[1], W.shape[0], conv.kernel_size,
                     stride=conv.stride, padding=conv.padding,
                     bias=(conv.bias is not None)).to(device)
    new.weight.data = torch.tensor(W, dtype=conv.weight.dtype, device=device)
    if bias is not None:
        new.bias.data = torch.tensor(bias, dtype=conv.bias.dtype, device=device)
    return new, idx


def prune_bn(bn, keep_idx, device="cpu"):
    new = nn.BatchNorm2d(len(keep_idx)).to(device)
    new.weight.data  = bn.weight.data[keep_idx].clone().to(device)
    new.bias.data    = bn.bias.data[keep_idx].clone().to(device)
    new.running_mean = bn.running_mean[keep_idx].clone().to(device)
    new.running_var  = bn.running_var[keep_idx].clone().to(device)
    return new


def update_next_conv_in(keep_in, conv, device="cpu"):
    keep_in = np.asarray(keep_in, dtype=int)
    if keep_in.ndim == 0:
        keep_in = np.arange(int(keep_in))
    keep_in = keep_in[(0 <= keep_in) & (keep_in < conv.in_channels)]
    if len(keep_in) == 0:
        keep_in = np.array([0])

    new = nn.Conv2d(len(keep_in), conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    bias=(conv.bias is not None)).to(device)
    new.weight.data = conv.weight.data[:, keep_in].clone().to(device)
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    return new


def rebuild_fc(model, fc_attr, in_shape=(1, 3, 32, 32), device="cpu"):
    """Probe the FC input size with a GPU dummy forward pass, then reinitialise."""
    model.eval()
    size = {}

    def hook(_, inp):
        size["n"] = inp[0].shape[1]

    h = getattr(model, fc_attr).register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(torch.zeros(*in_shape, device=device))
    except Exception:
        pass
    h.remove()

    if "n" not in size:
        raise RuntimeError(f"Could not determine input size for '{fc_attr}'")

    old = getattr(model, fc_attr)
    new = nn.Linear(size["n"], old.out_features).to(device)
    m   = min(old.in_features, size["n"])
    new.weight.data[:, :m] = old.weight.data[:, :m]
    new.bias.data = old.bias.data.clone()
    setattr(model, fc_attr, new)
    return model


# ── Evaluation ────────────────────────────────────────────────

def eval_acc(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def compute_ece(model, loader, n_bins=15, device="cpu"):
    model.eval()
    confs, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            prob   = F.softmax(model(x), dim=1)
            c, p   = prob.max(1)
            confs.append(c); preds.append(p); labels.append(y)
    confs  = torch.cat(confs)
    preds  = torch.cat(preds)
    labels = torch.cat(labels)
    ece    = 0.0
    for lo in torch.linspace(0, 1, n_bins + 1)[:-1]:
        hi   = lo + 1.0 / n_bins
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() > 0:
            ece += (mask.sum().float() / labels.size(0)) * \
                   torch.abs((preds[mask] == labels[mask]).float().mean()
                              - confs[mask].mean())
    return ece.item()


# ── FC fine-tuning with AMP ───────────────────────────────────

def fine_tune_fc(model, fc_attrs, train_loader, val_loader,
                 epochs=200, lr=1e-4, patience=10, device="cpu",
                 verbose_every=0):
    """
    Freeze conv layers, fine-tune *fc_attrs* only.
    Uses AMP (FP16) on CUDA for ~2× throughput.
    """
    use_amp = str(device).startswith("cuda")
    amp_device = "cuda" if use_amp else "cpu"

    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for attr in fc_attrs:
        for p in getattr(model, attr).parameters():
            p.requires_grad = True
            trainable.append(p)

    criterion  = nn.CrossEntropyLoss().to(device)
    optimizer  = optim.Adam(trainable, lr=lr)
    scaler     = GradScaler(amp_device, enabled=use_amp)
    best_acc   = 0.0
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        acc = eval_acc(model, val_loader, device)
        if acc > best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if verbose_every and ((ep + 1) % verbose_every == 0 or ep == 0):
            avg_loss = running_loss / max(1, seen)
            print(f"    fine-tune epoch {ep + 1:03d}/{epochs}  "
                  f"loss={avg_loss:.4f}  val={acc:.2f}  "
                  f"best={best_acc:.2f}  no_improve={no_improve}/{patience}",
                  flush=True)
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_acc, ep + 1


# ── HRank: GPU SVD ────────────────────────────────────────────

def hrank_scores(model, loader, device="cpu", num_batches=5, svd_thresh=0.01):
    """
    Average feature-map rank per filter using GPU SVD.
    Feature maps are kept on GPU throughout; only final rank arrays come back as numpy.
    """
    model.eval()
    fmaps = {}
    hooks = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            def _hook(_, __, out, n=name):
                # store on GPU, detach from graph
                fmaps.setdefault(n, []).append(out.detach())
            hooks.append(mod.register_forward_hook(_hook))

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(x.to(device, non_blocking=True))

    for h in hooks:
        h.remove()

    scores = {}
    for name, fmap_list in fmaps.items():
        fmap = torch.cat(fmap_list, dim=0)          # (B, C, H, W) on GPU
        B, C, H, W = fmap.shape
        ranks = []
        for c in range(C):
            mat = fmap[:, c].reshape(B, H * W)      # (B, H*W) still on GPU
            if H * W == 1:
                ranks.append(1); continue
            # GPU SVD – much faster than CPU for large B
            s = torch.linalg.svd(mat, full_matrices=False).S
            ranks.append(int((s > svd_thresh * s[0]).sum().item()))
        scores[name] = np.array(ranks, dtype=np.float32)
        del fmap                                     # free GPU memory

    torch.cuda.empty_cache()
    return scores
