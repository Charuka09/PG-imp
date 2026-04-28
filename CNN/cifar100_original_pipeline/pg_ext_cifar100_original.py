
import os
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import random
import time
import csv
from pathlib import Path

# ----------------------------
# Default Config (override via CLI)
# ----------------------------
models = ["conv2net", "conv6net", "lenet"]
layers_dict = {
    "conv2net": ['pool', 'fc'],
    "conv6net": ['pool1', 'pool2', 'pool3', 'fc1', 'fc2'],
    "lenet": ['fc1', 'fc2', 'fc3']
}

max_samples = 50
alpha_best, t_best = 0.5, 0.5
sigma = 0.1
chunk_size = 50


# ----------------------------
# PG1 function
# ----------------------------
def diffusion_pg1_safe(A, alpha=0.5, t=1, eps=1e-10, jitter=1e-6):
    A = (A + A.T) / 2
    A = np.where(A == 0, jitter, A)
    D = np.diag(A.sum(axis=1))
    diag_pow = np.power(np.diag(D), alpha)
    diag_pow = np.where(diag_pow == 0, eps, diag_pow)
    D_alpha = np.diag(1.0 / diag_pow)
    K = D_alpha @ A @ D_alpha
    M = K / (K.sum(axis=1, keepdims=True) + eps)
    if np.isnan(M).any() or np.isinf(M).any():
        M = np.nan_to_num(M, nan=jitter, posinf=jitter, neginf=jitter)
    if M.shape[0] == 1:
        return np.array([1.0])
    eigvals, eigvecs = eigh(M)
    idx = np.argsort(eigvals)[::-1]
    if len(idx) < 2:
        return eigvecs[:, idx[0]]
    pg1 = eigvecs[:, idx[1]]
    return (eigvals[idx[1]] ** t) * pg1


def load_and_prepare_activations(file_list, layer_name):
    activations = []
    for f in file_list:
        data = np.load(f, allow_pickle=True)
        activations.append(data.item()['activations'][layer_name].reshape(-1))
    return np.stack(activations)


def compute_pg_for_class_layer(model_name, dataset_name, cls, layer_name, base_load_dir, output_base,
                               max_samples=50, sigma=0.1, alpha=0.5, t=0.5):
    start = time.time()
    class_dir = Path(base_load_dir) / model_name / f"class_{cls}"
    if not class_dir.exists():
        return {"status": "skip", "cls": cls, "layer": layer_name, "time": 0.0}

    files = [str(p) for p in class_dir.glob("*.pt")]
    if len(files) == 0:
        return {"status": "skip", "cls": cls, "layer": layer_name, "time": 0.0}

    # Separate correct/incorrect files based on 'correct' flag stored in torch .pt
    # To keep original behavior, we load torch and then filter.
    import torch
    selected = []
    random.shuffle(files)
    for f in files:
        d = torch.load(f, map_location="cpu")
        is_corr = bool(d.get("correct", False))
        if (dataset_name == "correct" and is_corr) or (dataset_name == "incorrect" and (not is_corr)):
            selected.append(f)
        if len(selected) >= max_samples:
            break

    if len(selected) < 2:
        return {"status": "skip", "cls": cls, "layer": layer_name, "time": 0.0}

    # Build X: (samples, units)
    vecs = []
    for f in selected:
        d = torch.load(f, map_location="cpu")
        v = d["activations"][layer_name].detach().cpu().numpy().reshape(-1)
        vecs.append(v)
    X = np.stack(vecs, axis=0)   # (samples, units)
    X = X.T                      # (units, samples)

    A = cosine_similarity(X)     # (units, units)
    A = np.exp(- (1.0 - A)**2 / (2.0 * sigma**2))
    A += 1e-6
    np.fill_diagonal(A, A.diagonal() + 1e-10)

    pg1 = diffusion_pg1_safe(A, alpha=alpha, t=t)

    save_dir = Path(output_base) / model_name / dataset_name / "pg1_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{cls}_{layer_name}_pg1.npy"
    np.save(out_path, pg1)

    return {"status": "ok", "cls": cls, "layer": layer_name, "time": time.time() - start}


def run_pg(base_load_dir, output_base, num_classes=100, n_jobs=-1,
           max_samples=50, sigma=0.1, alpha=0.5, t=0.5):
    os.makedirs(output_base, exist_ok=True)

    model_list = models if args.model=="all" else [args.model]
    for model_name in model_list:
        layers = layers_dict[model_name]
        for dataset_name in ["correct", "incorrect"]:
            for layer_name in layers:
                results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(compute_pg_for_class_layer)(model_name, dataset_name, cls, layer_name,
                                                       base_load_dir, output_base,
                                                       max_samples=max_samples, sigma=sigma, alpha=alpha, t=t)
                    for cls in range(num_classes)
                )

                times = [r["time"] for r in results if r["status"] == "ok"]
                skipped = sum(1 for r in results if r["status"] != "ok")
                print(f"[{model_name}] {dataset_name} {layer_name}: ok={len(times)} skip={skipped} avg_time={np.mean(times) if times else 0:.2f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_load_dir", type=str, default="./activations")
    ap.add_argument("--output_base", type=str, default="./pg_multi_models_cifar100")
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--model", default="all", choices=["all","conv2net","conv6net","lenet"])
    args = ap.parse_args()

    run_pg(args.base_load_dir, args.output_base, num_classes=args.num_classes, n_jobs=args.n_jobs,
           max_samples=args.max_samples, sigma=args.sigma, alpha=args.alpha, t=args.t)
