
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import torch

"""
Compute and save functional connectivity maps (cosine similarity) for correct and incorrect samples
per class for multiple models and layers (CIFAR-100 version).
"""

models = ["conv2net", "conv6net", "lenet"]
layers_dict = {
    "conv2net": ['pool', 'fc'],
    "conv6net": ['pool1', 'pool2', 'pool3', 'fc1', 'fc2'],
    "lenet": ['fc1', 'fc2', 'fc3']
}

max_samples = 50
sigma = 0.1
eps = 1e-6

def cosine_similarity(A):
    # A: (units, samples)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return A @ A.T


def load_vectors(act_dir, cls, layer, want_correct=True, max_samples=50):
    cls_dir = Path(act_dir) / f"class_{cls}"
    if not cls_dir.exists():
        return None
    files = list(cls_dir.glob("*.pt"))
    random.shuffle(files)
    vecs = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        is_corr = bool(d.get("correct", False))
        if (want_correct and is_corr) or ((not want_correct) and (not is_corr)):
            v = d["activations"][layer].detach().cpu().numpy().reshape(-1)
            vecs.append(v)
        if len(vecs) >= max_samples:
            break
    if len(vecs) < 2:
        return None
    X = np.stack(vecs, axis=0).T  # (units, samples)
    return X


def run_cosine_maps(base_load_dir="./activations", save_dir="./w_fc_maps_dual_cifar100", num_classes=100):
    os.makedirs(save_dir, exist_ok=True)

    for model_name in models:
        act_root = Path(base_load_dir) / model_name
        for layer in layers_dict[model_name]:
            out_layer_dir = Path(save_dir) / model_name / layer
            out_layer_dir.mkdir(parents=True, exist_ok=True)

            for cls in range(num_classes):
                Xc = load_vectors(act_root, cls, layer, want_correct=True, max_samples=max_samples)
                Xi = load_vectors(act_root, cls, layer, want_correct=False, max_samples=max_samples)

                for tag, X in [("correct", Xc), ("incorrect", Xi)]:
                    if X is None:
                        continue
                    A = cosine_similarity(X)
                    A = np.exp(- (1.0 - A)**2 / (2.0 * sigma**2))
                    A = A + eps

                    np.save(out_layer_dir / f"class_{cls}_{tag}.npy", A)

                    # quick png
                    plt.figure(figsize=(4,4))
                    plt.imshow(A, aspect="auto")
                    plt.title(f"{model_name} {layer} class {cls} {tag}")
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(out_layer_dir / f"class_{cls}_{tag}.png", dpi=200)
                    plt.close()

            print(f"Saved cosine maps for {model_name} layer {layer} -> {out_layer_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_load_dir", type=str, default="./activations")
    ap.add_argument("--save_dir", type=str, default="./w_fc_maps_dual_cifar100")
    ap.add_argument("--num_classes", type=int, default=100)
    args = ap.parse_args()
    run_cosine_maps(args.base_load_dir, args.save_dir, num_classes=args.num_classes)
