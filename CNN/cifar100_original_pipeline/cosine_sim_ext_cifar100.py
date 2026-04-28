import torch
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

models = ["conv2net", "conv6net", "lenet"]
layers_dict = {
    "conv2net": ['pool', 'fc'],
    "conv6net": ['pool1', 'pool2', 'pool3', 'fc1', 'fc2'],
    "lenet": ['fc1', 'fc2']
}

def compute_cosine_matrix_full_chunked(x, chunk_size=256):
    units = x.shape[0]
    cos_mat = np.zeros((units, units), dtype=np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

    for i_start in range(0, units, chunk_size):
        i_end = min(i_start + chunk_size, units)
        x_row = x[i_start:i_end][:, None, :]
        for j_start in range(0, units, chunk_size):
            j_end = min(j_start + chunk_size, units)
            x_col = x[j_start:j_end][None, :, :]
            cos_mat[i_start:i_end, j_start:j_end] = np.sum(x_row * x_col, axis=2)
    return cos_mat

def plot_affinity_matrix(W, class_id, layer_name, dataset_type, matrix_type, save_path):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(W, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"CIFAR100 | class {class_id} | {layer_name} | {dataset_type} | {matrix_type}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    ap.add_argument("--base_load_dir", default="./activations_cifar100")
    ap.add_argument("--save_dir", default="./w_fc_maps_dual_cifar100")
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--chunk_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    layers = layers_dict[args.model]

    for class_id in range(100):
        class_dir = os.path.join(args.base_load_dir, args.model, f"class_{class_id}")
        if not os.path.exists(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(".pt")]

        correct_files, incorrect_files = [], []
        for f in files:
            data = torch.load(os.path.join(class_dir, f))
            if data['correct']:
                if len(correct_files) < args.max_samples:
                    correct_files.append(data)
            else:
                if len(incorrect_files) < args.max_samples:
                    incorrect_files.append(data)

        for dataset_type, file_list in zip(["correct", "incorrect"], [correct_files, incorrect_files]):
            if not file_list:
                continue

            for layer_name in layers:
                acts_samples = np.stack([d['activations'][layer_name].flatten().numpy() for d in file_list])

                cos_mat_samples = compute_cosine_matrix_full_chunked(acts_samples, chunk_size=args.chunk_size)
                W_samples = np.exp(-(1 - cos_mat_samples) ** 2 / (2 * 0.1 ** 2))

                save_path = os.path.join(args.save_dir, f"{args.model}_class{class_id}_{layer_name}_{dataset_type}_samples.png")
                plot_affinity_matrix(W_samples, class_id, layer_name, dataset_type, "sample_by_sample", save_path)

        print(f"[{args.model}] class {class_id}: cosine maps saved (if samples exist)")

if __name__ == "__main__":
    main()
