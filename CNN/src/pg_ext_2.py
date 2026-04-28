import torch
import os
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import random
import time
import csv

# ----------------------------
# Config
# ----------------------------
models = ["conv2net", "conv6net", "lenet"]
layers_dict = {
    "conv2net": ['pool', 'fc'],
    "conv6net": ['pool1', 'pool2', 'pool3', 'fc1', 'fc2'],
    "lenet": ['fc1', 'fc2']
}

base_load_dir = "/home/charuka09/Documents/postPhD/mindula/icml/activations"
output_base = "/home/charuka09/Documents/postPhD/mindula/icml/pg_multi_models_2"
os.makedirs(output_base, exist_ok=True)

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
        M += jitter * np.random.rand(*M.shape)
    if M.shape[0] == 1:
        return np.array([1.0])
    eigvals, eigvecs = eigh(M)
    idx = np.argsort(eigvals)[::-1]
    pg1 = eigvecs[:, idx[1]]
    return (eigvals[idx[1]]**t) * pg1

# ----------------------------
# Chunked cosine similarity
# ----------------------------
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

# ----------------------------
# Process PG1 per model/class/layer + TIME TRACKER
# ----------------------------
def process_model_class_layer(model_name, dataset_name, class_id, layer_name):
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
            acts_stack = data['activations'][layer_name].flatten()[None, :]  # add batch dim
            break  # only 1 sample needed

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
    print(f"✔ {model_name} | {dataset_name} | class {class_id} | {layer_name} done in {end_t-start_t:.2f}s")
    return {"class_id": class_id, "layer": layer_name, "time": end_t - start_t, "units": units, "status": "ok"}

# ----------------------------
# Main loop with MODEL + TOTAL timing
# ----------------------------
summary = {}
total_start = time.time()

for model_name in models:
    model_start = time.time()
    layers = layers_dict[model_name]
    summary[model_name] = {}

    for dataset_name in ["correct", "incorrect"]:
        summary[model_name][dataset_name] = {}

        for layer_name in layers:
            results = Parallel(n_jobs=-1)(
                delayed(process_model_class_layer)(model_name, dataset_name, cls, layer_name)
                for cls in range(10)
            )

            times = [r["time"] for r in results if r["status"] == "ok"]
            skipped = sum(1 for r in results if r["status"] != "ok")

            summary[model_name][dataset_name][layer_name] = {
                "avg_time": np.mean(times) if times else 0,
                "max_time": np.max(times) if times else 0,
                "min_time": np.min(times) if times else 0,
                "skipped": skipped
            }

    summary[model_name]["model_time"] = time.time() - model_start

total_end = time.time()

# ----------------------------
# Pretty Summary
# ----------------------------
print("\n==========================")
print("PG1 COMPUTE SUMMARY")
print("==========================")

for model_name, dsets in summary.items():
    print(f"\n=== Model: {model_name} ===")
    print(f"Model compute time: {dsets['model_time']:.2f} sec")

    for dataset_name, layers_info in dsets.items():
        if dataset_name == "model_time":
            continue
        print(f"\n  Dataset: {dataset_name}")
        for layer_name, stats in layers_info.items():
            print(f"    Layer {layer_name}: avg={stats['avg_time']:.3f}s, "
                  f"min={stats['min_time']:.3f}s, max={stats['max_time']:.3f}s, "
                  f"skipped {stats['skipped']}/10")

print(f"\nTotal runtime: {total_end - total_start:.2f} sec")

import csv
# ----------------------------
# Save summary to CSV
# ----------------------------
summary_csv_path = os.path.join(output_base, "pg1_compute_summary.csv")

with open(summary_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Dataset", "Layer", "Avg_Time(s)", "Min_Time(s)", "Max_Time(s)", "Skipped", "Units"])
    
    for model_name, dsets in summary.items():
        for dataset_name, layers_info in dsets.items():
            if dataset_name == "model_time":
                continue
            for layer_name, stats in layers_info.items():
                units = stats.get("units", "N/A")  # in case units not saved
                writer.writerow([
                    model_name,
                    dataset_name,
                    layer_name,
                    stats["avg_time"],
                    stats["min_time"],
                    stats["max_time"],
                    stats["skipped"],
                    units
                ])
