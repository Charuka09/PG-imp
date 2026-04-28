import torch
import os
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import random
import time
import argparse

models = ["conv2net", "conv6net", "lenet"]
layers_dict = {
    "conv2net": ['pool', 'fc'],
    "conv6net": ['pool1', 'pool2', 'pool3', 'fc1', 'fc2'],
    "lenet": ['fc1', 'fc2']
}

# ----------------------------
# PG1 function (original)
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

def cosine_similarity_chunked(X, chunk_size=50):
    units = X.shape[0]
    A = np.zeros((units, units))
    for i in range(0, units, chunk_size):
        i_end = min(i + chunk_size, units)
        X_i = X[i:i_end]
        for j in range(0, units, chunk_size):
            j_end = min(j + chunk_size, units)
            X_j = X[j:j_end]
            A[i:i_end, j:j_end] = cosine_similarity(X_i, X_j)
    return A

def process_model_class_layer(base_load_dir, output_base, model_name, dataset_name, class_id, layer_name,
                              sigma=0.1, alpha_best=0.5, t_best=0.5, chunk_size=50):
    class_dir = os.path.join(base_load_dir, model_name, f"class_{class_id}")
    if not os.path.exists(class_dir):
        return {"class_id": class_id, "layer": layer_name, "time": 0, "status": "missing"}

    files = [f for f in os.listdir(class_dir) if f.endswith(".pt")]
    random.shuffle(files)

    acts_stack = None
    for f in files:
        data = torch.load(os.path.join(class_dir, f))
        correct = data.get('correct', False)
        if (dataset_name == "correct" and correct) or (dataset_name == "incorrect" and not correct):
            # ORIGINAL: only ONE sample per class per split
            acts_stack = data['activations'][layer_name].flatten()[None, :]
            break

    if acts_stack is None:
        return {"class_id": class_id, "layer": layer_name, "time": 0, "status": "no_samples"}

    units = acts_stack.shape[1]
    start_t = time.time()

    if units == 1:
        A_layer = np.array([[1.0]])
    else:
        A_layer = cosine_similarity_chunked(acts_stack.T, chunk_size=chunk_size)
        A_layer = np.exp(- (1 - A_layer)**2 / (2 * sigma**2))
        A_layer += 1e-6
        A_layer = (A_layer + A_layer.T) / 2
        np.fill_diagonal(A_layer, A_layer.diagonal() + 1e-10)

    pg1_layer = diffusion_pg1_safe(A_layer, alpha=alpha_best, t=t_best)

    save_dir = os.path.join(output_base, model_name, dataset_name, "pg1_data")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{class_id}_{layer_name}_pg1.npy"), pg1_layer)

    end_t = time.time()
    return {"class_id": class_id, "layer": layer_name, "time": end_t - start_t, "units": units, "status": "ok"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    ap.add_argument("--base_load_dir", default="./activations_cifar100")
    ap.add_argument("--output_base", default="./pg_cifar100")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--chunk_size", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.output_base, exist_ok=True)
    layers = layers_dict[args.model]

    for dataset_name in ["correct", "incorrect"]:
        for layer_name in layers:
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_model_class_layer)(
                    args.base_load_dir, args.output_base, args.model, dataset_name, cls, layer_name,
                    sigma=args.sigma, alpha_best=args.alpha, t_best=args.t, chunk_size=args.chunk_size
                )
                for cls in range(100)
            )
            ok = sum(1 for r in results if r["status"] == "ok")
            skipped = len(results) - ok
            times = [r["time"] for r in results if r["status"] == "ok"]
            print(f"[{args.model}] {dataset_name} {layer_name}: ok={ok} skipped={skipped} mean_time={np.mean(times) if times else 0:.2f}s")

    print(f"PG done. Saved under: {args.output_base}/{args.model}/...")

if __name__ == "__main__":
    main()
