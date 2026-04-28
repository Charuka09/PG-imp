import os
import math
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from scipy.linalg import eigh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset: AG News (4 classes)
# ----------------------------
def load_agnews(tokenizer, max_length=128):
    ds = load_dataset("ag_news")  # train/test splits
    # label mapping is 0..3 already

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    return ds


# ----------------------------
# Basic training loop (simple, fast, no Trainer)
# ----------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch["labels"].to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)

@torch.no_grad()
def compute_ece(model, loader, device, n_bins=15):
    """
    n_bins=15 is a de facto standard in modern ML.
    Multiclass Expected Calibration Error (ECE), using max softmax confidence.
    """
    model.eval()
    confs = []
    corrects = []
    total = 0

    for batch in tqdm(loader, desc="ece", leave=False):
        labels = batch["labels"].to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        confs.append(conf.detach().cpu())
        corrects.append((pred == labels).detach().cpu())
        total += labels.size(0)

    confs = torch.cat(confs)
    corrects = torch.cat(corrects).float()

    ece = 0.0
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if mask.any():
            bin_acc = corrects[mask].mean().item()
            bin_conf = confs[mask].mean().item()
            bin_frac = mask.float().mean().item()
            ece += bin_frac * abs(bin_acc - bin_conf)

    return float(ece)


def get_pruned_pct_from_importance(importance_npz, target_layers, threshold):
    """
    Returns the overall % of pruned FFN neurons across all target layers.
    Also returns per-layer pruned % dict for logging.
    """
    imp_data = np.load(importance_npz)
    total = 0
    kept = 0
    per_layer = {}
    for l in target_layers:
        key = f"layer{l}"
        if key not in imp_data:
            continue
        imp = imp_data[key]
        thr_value = np.percentile(imp, threshold * 100)
        keep = imp > thr_value
        total += keep.size
        kept += int(keep.sum())
        per_layer[l] = 100.0 * (1.0 - keep.mean())
    if total == 0:
        return 0.0, per_layer
    pruned_pct = 100.0 * (1.0 - (kept / total))
    return float(pruned_pct), per_layer


def append_row_csv(csv_path, row_dict, header):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
# ----------------------------
# Activation extraction (FFN intermediate, per-layer)
# We save per-sample activations at [CLS] position: (3072,)
# ----------------------------
def extract_activations_correct_incorrect(
    model,
    tokenizer,
    dataset,
    device,
    save_root,
    target_layers,         # list of encoder layer indices, e.g. [0, 5, 11]
    max_per_class=100,     # collect up to 100 correct + 100 incorrect per class
    batch_size=32,
):
    os.makedirs(save_root, exist_ok=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    # we manually batch using simple slicing to keep dependencies light

    n_classes = 4
    counters = {c: {"correct": 0, "incorrect": 0} for c in range(n_classes)}

    # hook containers: layer_idx -> activation tensor for the last forward pass
    acts = {l: None for l in target_layers}
    hooks = []

    def make_hook(layer_idx):
        # hook on the intermediate dense output (before GELU in HF BERT impl),
        # we apply GELU ourselves for consistency.
        def hook_fn(module, inp, out):
            # out: [B, T, 3072]
            acts[layer_idx] = F.gelu(out).detach().cpu()
        return hook_fn

    for l in target_layers:
        inter_dense = model.bert.encoder.layer[l].intermediate.dense
        hooks.append(inter_dense.register_forward_hook(make_hook(l)))

    model.to(device)
    model.eval()

    # iterate through dataset sequentially until all classes filled
    idx = 0
    pbar = tqdm(total=len(dataset), desc="extract", leave=True)

    while idx < len(dataset):
        # stop condition: all classes have enough correct and incorrect
        done = all(
            counters[c]["correct"] >= max_per_class and counters[c]["incorrect"] >= max_per_class
            for c in range(n_classes)
        )
        if done:
            break

        batch_items = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]
        idx += len(batch_items)
        pbar.update(len(batch_items))

        batch = collator(batch_items)
        labels = batch["labels"].clone()

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu()

        labels_cpu = labels.cpu()
        correct_mask = (preds == labels_cpu)

        # per-sample saving (like your CIFAR code)
        for bi in range(len(labels_cpu)):
            cls = int(labels_cpu[bi].item())
            is_correct = bool(correct_mask[bi].item())

            # skip if quota reached
            if is_correct and counters[cls]["correct"] >= max_per_class:
                continue
            if (not is_correct) and counters[cls]["incorrect"] >= max_per_class:
                continue

            sample_dir = os.path.join(save_root, f"class_{cls}")
            os.makedirs(sample_dir, exist_ok=True)

            # collect per-layer activations (CLS token only)
            sample_acts = {}
            for l in target_layers:
                a = acts[l]  # [B,T,3072] on CPU
                if a is None:
                    raise RuntimeError("Hook did not capture activations. Check layer/module names.")
                sample_acts[f"layer{l}_ffn"] = a[bi, 0, :].contiguous()  # CLS position

            payload = {
                "activations": sample_acts,
                "correct": is_correct,
                "label": cls,
            }
            fname = f"idx{idx}_b{bi}_{'correct' if is_correct else 'incorrect'}.pt"
            torch.save(payload, os.path.join(sample_dir, fname), pickle_protocol=4)

            if is_correct:
                counters[cls]["correct"] += 1
            else:
                counters[cls]["incorrect"] += 1

    pbar.close()
    for h in hooks:
        h.remove()

    print("Extraction complete. Counters:")
    for c in range(n_classes):
        print(f"  class {c}: {counters[c]}")


# ----------------------------
# PG1 (diffusion) helpers
# ----------------------------
def diffusion_pg1_safe(A, alpha=0.5, t=0.5, eps=1e-10, jitter=1e-6):
    A = (A + A.T) / 2
    A = np.where(A == 0, jitter, A)
    D = np.diag(A.sum(axis=1))
    diag_pow = np.power(np.diag(D), alpha)
    diag_pow = np.where(diag_pow == 0, eps, diag_pow)
    D_alpha = np.diag(1.0 / diag_pow)
    K = D_alpha @ A @ D_alpha
    M = K / (K.sum(axis=1, keepdims=True) + eps)

    if np.isnan(M).any() or np.isinf(M).any():
        M += jitter * np.random.rand(*M.shape)

    if M.shape[0] == 1:
        return np.array([1.0], dtype=np.float32)

    eigvals, eigvecs = eigh(M)
    idx = np.argsort(eigvals)[::-1]
    # second eigenvector
    pg1 = eigvecs[:, idx[1]]
    return (eigvals[idx[1]] ** t) * pg1


def cosine_similarity_chunked(X, chunk_size=256):
    # X: (units, samples)
    units = X.shape[0]
    A = np.zeros((units, units), dtype=np.float32)
    # normalize rows
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    for i in range(0, units, chunk_size):
        i_end = min(i + chunk_size, units)
        Xi = X[i:i_end][:, None, :]  # (bi,1,S)
        for j in range(0, units, chunk_size):
            j_end = min(j + chunk_size, units)
            Xj = X[j:j_end][None, :, :]  # (1,bj,S)
            A[i:i_end, j:j_end] = np.sum(Xi * Xj, axis=2)
    return A

# ----------------------------
# Plotting Helpers
# ----------------------------


def plot_affinity_matrix(W, title, save_path):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(W, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title, fontsize=14)
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def cosine_similarity_samples_chunked(X, chunk_size=256):
    """
    X: (samples, units)
    returns: (samples, samples) cosine similarity
    """
    n = X.shape[0]
    C = np.zeros((n, n), dtype=np.float32)

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        Xi = Xn[i:i_end]  # (bi, units)
        for j in range(0, n, chunk_size):
            j_end = min(j + chunk_size, n)
            Xj = Xn[j:j_end]  # (bj, units)
            C[i:i_end, j:j_end] = Xi @ Xj.T
    return C

# ----------------------------
# Compute PG1 per (class, layer, correct/incorrect)
# ----------------------------
def compute_pg1_from_saved_activations(
    act_root,
    out_root,
    imp_path,
    target_layers,
    max_samples=50,
    sigma=0.1,
    chunk_size=256,
    alpha=0.5,
    t=0.5,
):
    os.makedirs(out_root, exist_ok=True)
    n_classes = 4

    for cls in range(n_classes):
        class_dir = os.path.join(act_root, f"class_{cls}")
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(".pt")]
        # split correct vs incorrect, cap at max_samples each
        correct_data, incorrect_data = [], []
        for f in files:
            d = torch.load(os.path.join(class_dir, f), map_location="cpu")
            if d["correct"] and len(correct_data) < max_samples:
                correct_data.append(d)
            if (not d["correct"]) and len(incorrect_data) < max_samples:
                incorrect_data.append(d)
            if len(correct_data) >= max_samples and len(incorrect_data) >= max_samples:
                break

        for split_name, data_list in [("correct", correct_data), ("incorrect", incorrect_data)]:
            if not data_list:
                continue

            for l in target_layers:
                key = f"layer{l}_ffn"
                # stack: samples x units(3072)
                X = np.stack([d["activations"][key].numpy() for d in data_list], axis=0)
                # units x samples for unit-by-unit connectivity
                Xu = X.T

                A = cosine_similarity_chunked(Xu, chunk_size=chunk_size)
                # Gaussian affinity on (1 - cosine)
                W = np.exp(-((1 - A) ** 2) / (2 * (sigma ** 2))).astype(np.float32)
                W = (W + W.T) / 2
                np.fill_diagonal(W, W.diagonal() + 1e-6)

                pg1 = diffusion_pg1_safe(W, alpha=alpha, t=t)

                # Stack activations: samples × units
                X = np.stack([d["activations"][key].numpy() for d in data_list], axis=0)

                # ---------- Sample-by-sample affinity ----------
                cos_s = cosine_similarity_samples_chunked(X, chunk_size=chunk_size)
                W_samples = np.exp(-((1 - cos_s) ** 2) / (2 * (sigma ** 2))).astype(np.float32)
                W_samples = (W_samples + W_samples.T) / 2
                np.fill_diagonal(W_samples, W_samples.diagonal() + 1e-6)

                plot_dir = os.path.join(out_root, "plots", split_name, "samples")
                os.makedirs(plot_dir, exist_ok=True)
                samples_plot_path = os.path.join(plot_dir, f"class{cls}_layer{l}_samples.png")
                units_plot_dir = os.path.join(out_root, f"class{cls}_layer{l}_units.png")
                plot_affinity_matrix(
                    W_samples,
                    title=f"AGNews | class {cls} | layer {l} | {split_name} | samples×samples",
                    save_path=samples_plot_path
                )

                imp = np.load(imp_path)[f"layer{l}"]
                top_idx = np.argsort(imp)[-200:]
                A_top = A[np.ix_(top_idx, top_idx)]
                plot_affinity_matrix(
                    A_top,
                    title=f"BERT | class {cls} | layer {l} | {split_name} | units×units",
                    save_path=units_plot_dir
                )

                save_dir = os.path.join(out_root, split_name, "pg1_data")
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f"{cls}_layer{l}_ffn_pg1.npy"), pg1)

        print(f"PG1 saved for class {cls} | correct={len(correct_data)} incorrect={len(incorrect_data)}")


# ----------------------------
# Build neuron importance from PG-diff via PCA (per layer)
# ----------------------------
def build_importance_from_pgdiff(pg_root, out_path, target_layers, n_classes=4, pca_components=4):
    """
    For each layer l:
      - load pg_correct - pg_incorrect for each class
      - PCA across classes
      - importance per neuron = abs(mean projection scores across top components)
    Saves: npz with importance per layer
    """
    importance = {}
    print("pg_root", pg_root)
    print("out_path", out_path)
    print("target_layers", target_layers)
    for l in target_layers:
        diffs = []
        for cls in range(n_classes):
            p_c = os.path.join(pg_root, "correct", "pg1_data", f"{cls}_layer{l}_ffn_pg1.npy")
            p_i = os.path.join(pg_root, "incorrect", "pg1_data", f"{cls}_layer{l}_ffn_pg1.npy")
            if not (os.path.exists(p_c) and os.path.exists(p_i)):
                continue
            diffs.append(np.load(p_c) - np.load(p_i))

        diffs = np.array(diffs)
        if diffs.size == 0:
            print(f"Skipping layer {l}: no PG diff data.")
            continue

        k = min(pca_components, diffs.shape[0], diffs.shape[1])
        pca = PCA(n_components=k)
        pca.fit(diffs)
        comps = pca.components_  # (k, units)

        # neuron importance: mean abs across components
        imp = np.mean(np.abs(comps), axis=0)
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
        importance[f"layer{l}"] = imp.astype(np.float32)

        print(f"✔ importance built for layer {l} (units={imp.shape[0]})")

    np.savez(out_path, **importance)
    print(f"Saved importance to: {out_path}")


# ----------------------------
# Structured FFN pruning on BERT
# Prune intermediate neurons by zeroing:
#   intermediate.dense weight rows/bias
#   output.dense weight cols
# (keeps shapes same; easier than surgery)
# ----------------------------
def apply_ffn_neuron_pruning(model, importance_npz, target_layers, threshold):
    imp_data = np.load(importance_npz)
    for l in target_layers:
        key = f"layer{l}"
        if key not in imp_data:
            continue
        imp = imp_data[key]  # (3072,)
        thr_value = np.percentile(imp, threshold * 100)
        keep = imp > thr_value
        if keep.sum() == 0:
            keep[np.argmax(imp)] = True  # keep at least one

        layer = model.bert.encoder.layer[l]
        inter = layer.intermediate.dense
        out = layer.output.dense

        keep_t = torch.tensor(keep, device=inter.weight.device, dtype=torch.bool)

        # Zero pruned neurons: intermediate rows + bias
        with torch.no_grad():
            inter.weight[~keep_t, :] = 0
            if inter.bias is not None:
                inter.bias[~keep_t] = 0

            # output.dense takes intermediate output -> hidden, so zero columns
            out.weight[:, ~keep_t] = 0

        pruned_pct = 100.0 * (1 - keep.mean())
        print(f"Layer {l}: pruned {pruned_pct:.2f}% FFN neurons (threshold={threshold})")

def apply_ffn_mask(model, target_layers, keep_masks):
    """
    keep_masks: dict {layer_idx: boolean array of shape (3072,)}
    """
    for l in target_layers:
        keep = keep_masks[l]
        layer = model.bert.encoder.layer[l]
        inter = layer.intermediate.dense
        out = layer.output.dense

        keep_t = torch.tensor(keep, device=inter.weight.device, dtype=torch.bool)

        with torch.no_grad():
            inter.weight[~keep_t, :] = 0
            if inter.bias is not None:
                inter.bias[~keep_t] = 0
            out.weight[:, ~keep_t] = 0

def random_pruning_masks(target_layers, units, prune_ratio, seed=42):
    rng = np.random.default_rng(seed)
    masks = {}
    for l in target_layers:
        keep = np.ones(units, dtype=bool)
        n_prune = int(prune_ratio * units)
        idx = rng.choice(units, size=n_prune, replace=False)
        keep[idx] = False
        masks[l] = keep
    return masks

def l1_pruning_masks(model, target_layers, prune_ratio):
    masks = {}
    for l in target_layers:
        W = model.bert.encoder.layer[l].intermediate.dense.weight.detach().cpu().numpy()
        scores = np.mean(np.abs(W), axis=1)   # per-neuron L1
        k = int(prune_ratio * len(scores))
        thresh = np.partition(scores, k)[k]
        masks[l] = scores > thresh
    return masks

def l2_pruning_masks(model, target_layers, prune_ratio):
    masks = {}
    for l in target_layers:
        W = model.bert.encoder.layer[l].intermediate.dense.weight.detach().cpu().numpy()
        scores = np.sqrt(np.mean(W**2, axis=1))
        k = int(prune_ratio * len(scores))
        thresh = np.partition(scores, k)[k]
        masks[l] = scores > thresh
    return masks

def build_importance_random(out_path, target_layers, units=3072, seed=42):
    """
    Random importance per neuron (higher = keep).
    Saves npz: layer{l} -> (units,) float32
    """
    rng = np.random.default_rng(seed)
    importance = {}
    for l in target_layers:
        importance[f"layer{l}"] = rng.random(units).astype(np.float32)
    np.savez(out_path, **importance)
    print(f"Saved RANDOM importance to: {out_path}")


def build_importance_l1(model, out_path, target_layers):
    """
    Magnitude baseline: per-neuron score from FFN weights.
    Score neuron j using both:
      - row j of intermediate.dense.weight
      - column j of output.dense.weight
    """
    importance = {}
    model.eval()
    with torch.no_grad():
        for l in target_layers:
            layer = model.bert.encoder.layer[l]
            W_in = layer.intermediate.dense.weight.detach().cpu().numpy()  # [3072, 768]
            W_out = layer.output.dense.weight.detach().cpu().numpy()       # [768, 3072]

            s_in = np.sum(np.abs(W_in), axis=1)     # [3072]
            s_out = np.sum(np.abs(W_out), axis=0)   # [3072]
            imp = s_in + s_out

            imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
            importance[f"layer{l}"] = imp.astype(np.float32)

    np.savez(out_path, **importance)
    print(f"Saved L1 importance to: {out_path}")


def build_importance_l2(model, out_path, target_layers):
    """
    Magnitude baseline: L2 per-neuron score from FFN weights.
    """
    importance = {}
    model.eval()
    with torch.no_grad():
        for l in target_layers:
            layer = model.bert.encoder.layer[l]
            W_in = layer.intermediate.dense.weight.detach().cpu().numpy()
            W_out = layer.output.dense.weight.detach().cpu().numpy()

            s_in = np.sqrt(np.sum(W_in * W_in, axis=1) + 1e-12)  # [3072]
            s_out = np.sqrt(np.sum(W_out * W_out, axis=0) + 1e-12)

            imp = s_in + s_out
            imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-12)
            importance[f"layer{l}"] = imp.astype(np.float32)

    np.savez(out_path, **importance)
    print(f"Saved L2 importance to: {out_path}")
# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["finetune", "extract", "pg", "importance", "prune_retrain", "sweep"])
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--out_dir", default="/home/charuka09/Documents/postPhD/mindula/icml/Bert/agnews/")
    ap.add_argument("--model_path", default="/home/charuka09/Documents/postPhD/mindula/icml/Bert/agnews/new/")
    ap.add_argument("--target_layers", default="0,5,11", help="comma-separated layer indices")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_per_class", type=int, default=100)
    ap.add_argument("--max_samples_pg", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--pca_k", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                help="comma-separated thresholds for sweep")
    ap.add_argument("--csv_name", default="sweep_results.csv",
                    help="CSV filename to write under the mode folder")
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--prune_method",
        choices=["pg", "random", "l1", "l2"],
        default="random"
    )
    args = ap.parse_args()

    if args.freeze_encoder:
        model_mode = "head"
    else:
        model_mode = "full"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_layers = [int(x) for x in args.target_layers.split(",")]
    if args.stage not in ["prune_retrain", "sweep"]:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(args.out_dir, f"{model_mode}")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = os.path.join(args.model_path, f"fine_tuned_model")
    act_dir = os.path.join(args.model_path, f"activations")
    pg_dir = os.path.join(args.model_path, f"pg")
    imp_path = os.path.join(args.model_path, f"importance_ffn.npz")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_agnews(tokenizer, max_length=args.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def make_loader(split, shuffle):
        return torch.utils.data.DataLoader(
            ds[split],
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
        )

    if args.stage == "finetune":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=4).to(device)
        train_loader = make_loader("train", True)
        test_loader = make_loader("test", False)
        # if args.freeze_encoder:
        #     # Freeze all BERT encoder params
        #     for p in model.bert.parameters():
        #         p.requires_grad = False
        #     # Train only the classifier head
        #     for p in model.classifier.parameters():
        #         p.requires_grad = True

        #     # IMPORTANT: optimizer should only see trainable params
        #     optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        # else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best = 0.0
        for ep in range(args.epochs):
            loss = train_epoch(model, train_loader, optimizer, device)
            acc = eval_acc(model, test_loader, device)
            print(f"[epoch {ep+1}] loss={loss:.4f} test_acc={acc:.4f}")
            if acc > best:
                best = acc
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
        print(f"Best test_acc={best:.4f}. Saved to {ckpt_dir}")

    elif args.stage == "extract":
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        # extract from TEST split for clean correctness measurement
        extract_activations_correct_incorrect(
            model=model,
            tokenizer=tokenizer,
            dataset=ds["test"],
            device=device,
            save_root=act_dir,
            target_layers=target_layers,
            max_per_class=args.max_per_class,
            batch_size=args.batch_size,
        )

    elif args.stage == "pg":
        compute_pg1_from_saved_activations(
            act_root=act_dir,
            out_root=pg_dir,
            imp_path=imp_path,
            target_layers=target_layers,
            max_samples=args.max_samples_pg,
            sigma=args.sigma,
            chunk_size=args.chunk_size,
            alpha=args.alpha,
            t=args.t,
        )

    elif args.stage == "importance":
        build_importance_from_pgdiff(
            pg_root=pg_dir,
            out_path=imp_path,
            target_layers=target_layers,
            n_classes=4,
            pca_components=args.pca_k,
        )

    elif args.stage == "prune_retrain":
        # load fine-tuned model, prune, then retrain a bit
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)

        print("Baseline accuracy:")
        base_acc = eval_acc(model, make_loader("test", False), device)
        print(f"  test_acc={base_acc:.4f}")

        apply_ffn_neuron_pruning(model, imp_path, target_layers, threshold=args.threshold)

        print("Post-prune (no retrain) accuracy:")
        pruned_acc = eval_acc(model, make_loader("test", False), device)
        print(f"  test_acc={pruned_acc:.4f}")

        # quick fine-tune to recover
        train_loader = make_loader("train", True)
        test_loader = make_loader("test", False)
        if args.freeze_encoder:
            for p in model.bert.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best = 0.0
        best_dir = os.path.join(out_dir, f"pruned_{model_mode}_thr{args.threshold}")
        os.makedirs(best_dir, exist_ok=True)

        for ep in range(max(1, args.epochs)):
            loss = train_epoch(model, train_loader, optimizer, device)
            acc = eval_acc(model, test_loader, device)
            print(f"[retrain epoch {ep+1}] loss={loss:.4f} test_acc={acc:.4f}")
            if acc > best:
                best = acc
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

        print(f"Best retrained acc={best:.4f}. Saved to {best_dir}")

    elif args.stage == "sweep":
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip() != ""]
        train_loader = make_loader("train", True)
        test_loader = make_loader("test", False)

        # CSV path is mode-specific
        csv_path = os.path.join(out_dir, args.csv_name)

        header = [
            "timestamp",
            "pruning_method",
            "model_mode",
            "model_name",
            "target_layers",
            "threshold",
            "pruned_pct_overall",
            "baseline_acc",
            "baseline_ece",
            "postprune_acc",
            "postprune_ece",
            "best_retrain_acc",
            "best_retrain_ece",
            "epochs",
            "lr",
            "seed",
            "per_layer_pruned_pct"
        ]

        # ---- baseline metrics (once) ----
        base_model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        baseline_acc = eval_acc(base_model, test_loader, device)
        baseline_ece = compute_ece(base_model, test_loader, device, n_bins=args.ece_bins)
        print(f"[BASELINE] acc={baseline_acc:.4f} ece={baseline_ece:.4f}")

        for thr in thresholds:
            print(f"\n=== SWEEP threshold={thr:.3f} | mode={model_mode} ===")

            # Fresh load for each threshold (like CNN: deepcopy model each run)
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)

            # --- choose importance file per method ---
            method_imp_path = os.path.join(args.model_path, f"importance_ffn_{args.prune_method}.npz")

            # Build importance once if missing
            if not os.path.exists(method_imp_path):
                if args.prune_method == "pg":
                    # must already have PG computed -> importance stage should have produced imp_path
                    if not os.path.exists(imp_path):
                        raise FileNotFoundError(f"PG importance not found at {imp_path}. Run --stage importance first.")
                    # just copy/reference the PG importance
                    method_imp_path = imp_path

                elif args.prune_method == "random":
                    build_importance_random(method_imp_path, target_layers, units=3072, seed=args.seed)

                elif args.prune_method == "l1":
                    tmp_model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
                    build_importance_l1(tmp_model, method_imp_path, target_layers)

                elif args.prune_method == "l2":
                    tmp_model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
                    build_importance_l2(tmp_model, method_imp_path, target_layers)

            # compute prune % for logging FROM THIS METHOD
            pruned_pct, per_layer = get_pruned_pct_from_importance(method_imp_path, target_layers, thr)

            # apply pruning (same function for all)
            apply_ffn_neuron_pruning(model, method_imp_path, target_layers, threshold=thr)

            # post-prune metrics
            post_acc = eval_acc(model, test_loader, device)
            post_ece = compute_ece(model, test_loader, device, n_bins=args.ece_bins)
            print(f"[POST-PRUNE] acc={post_acc:.4f} ece={post_ece:.4f} pruned={pruned_pct:.2f}%")

            # retrain setup (respect freeze_encoder)
            if args.freeze_encoder:
                for p in model.bert.parameters():
                    p.requires_grad = False
                for p in model.classifier.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            # retrain + keep best by test accuracy (same logic as your prune_retrain)
            best_acc = -1.0
            best_state = None
            for ep in range(max(1, args.epochs)):
                loss = train_epoch(model, train_loader, optimizer, device)
                acc = eval_acc(model, test_loader, device)
                print(f"[RETRAIN ep {ep+1}] loss={loss:.4f} acc={acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    # store best weights in-memory so sweep doesn't create tons of folders
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # restore best & compute ECE at best point
            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            best_ece = compute_ece(model, test_loader, device, n_bins=args.ece_bins)

            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_mode": model_mode,
                "pruning_method": args.prune_method,
                "model_name": args.model_name,
                "target_layers": args.target_layers,
                "threshold": thr,
                "pruned_pct_overall": pruned_pct,
                "baseline_acc": baseline_acc,
                "baseline_ece": baseline_ece,
                "postprune_acc": post_acc,
                "postprune_ece": post_ece,
                "best_retrain_acc": float(best_acc),
                "best_retrain_ece": float(best_ece),
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed,
                "per_layer_pruned_pct": str(per_layer),
            }

            append_row_csv(csv_path, row, header)
            print(f"appended -> {csv_path}")

if __name__ == "__main__":
    main()