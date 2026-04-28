import torch
import os
import numpy as np
'''
This is to get the functional connectivity matrices (cosine similarity) for correct and incorrect samples
separately per class for multiple models and layers... this matches the pg extraction setup.

'''
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Config
# -------------------
models = ["conv2net", "conv6net", "lenet"]
base_load_dir = "/home/charuka09/Documents/postPhD/mindula/icml/activations"
layers_dict = {
    "conv2net": ['pool', 'fc'],
    "conv6net": ['pool1', 'pool2', 'pool3', 'fc1', 'fc2'],
    "lenet": ['fc1', 'fc2']
}
max_samples = 50
sigma = 0.1
eps = 1e-6
chunk_size = 256

save_dir = "/home/charuka09/Documents/postPhD/mindula/icml/w_fc_maps_dual"
os.makedirs(save_dir, exist_ok=True)

# -------------------
# Class label mapping
# -------------------
cifar10_classes = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
mnist_classes = [str(i) for i in range(10)]

model_class_map = {
    "conv2net": cifar10_classes,
    "conv6net": cifar10_classes,
    "lenet": mnist_classes
}

# -------------------
# Cosine similarity function
# -------------------
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

# -------------------
# Plotting helper
# -------------------
def plot_affinity_matrix(W, class_label, layer_name, dataset_type, matrix_type, save_path):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(W, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"{dataset_type.capitalize()} | {class_label} | Layer {layer_name} | {matrix_type}", fontsize=20)
    
    # Set axis labels based on type
    if matrix_type == "samples":
        plt.xlabel("Samples", fontsize=20)
        plt.ylabel("Samples", fontsize=20)
    elif matrix_type == "units":
        plt.xlabel("Units", fontsize=20)
        plt.ylabel("Units", fontsize=20)
    else:
        plt.xlabel(matrix_type.capitalize(), fontsize=20)
        plt.ylabel(matrix_type.capitalize(), fontsize=20)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------
# Main loop
# -------------------
for model_name in models:
    model_dir = os.path.join(base_load_dir, model_name)
    layers = layers_dict[model_name]
    class_labels = model_class_map[model_name]

    for class_id, class_label in enumerate(class_labels):
        class_dir = os.path.join(model_dir, f"class_{class_id}")
        if not os.path.exists(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(".pt")]

        correct_files, incorrect_files = [], []
        for f in files:
            data = torch.load(os.path.join(class_dir, f))
            if data['correct']:
                if len(correct_files) < max_samples:
                    correct_files.append(data)
            else:
                if len(incorrect_files) < max_samples:
                    incorrect_files.append(data)

        for dataset_type, file_list in zip(["correct", "incorrect"], [correct_files, incorrect_files]):
            if not file_list:
                continue

            for layer_name in layers:
                # Stack activations: samples × units
                acts_samples = np.stack([d['activations'][layer_name].flatten() for d in file_list])

                # -------- Sample-by-sample --------
                cos_mat_samples = compute_cosine_matrix_full_chunked(acts_samples)
                W_samples = np.exp(- (1 - cos_mat_samples)**2 / (2 * sigma**2))
                W_samples = (W_samples + W_samples.T)/2 + eps * np.eye(W_samples.shape[0])

                # -------- Unit-by-unit (PG1 style) --------
                acts_units = acts_samples.T  # units × samples
                cos_mat_units = compute_cosine_matrix_full_chunked(acts_units)
                W_units = np.exp(- (1 - cos_mat_units)**2 / (2 * sigma**2))
                W_units = (W_units + W_units.T)/2 + eps * np.eye(W_units.shape[0])

                # -------- Save folders --------
                for mat_type, W in zip(["samples", "units"], [W_samples, W_units]):
                    save_class_dir = os.path.join(save_dir, model_name, dataset_type, mat_type)
                    os.makedirs(save_class_dir, exist_ok=True)
                    save_path = os.path.join(save_class_dir, f"{class_label}_{layer_name}.png")
                    plot_affinity_matrix(W, class_label, layer_name, dataset_type, mat_type, save_path)

        print(f"✅ Done {model_name} class {class_label} | Correct: {len(correct_files)}, Incorrect: {len(incorrect_files)}")


