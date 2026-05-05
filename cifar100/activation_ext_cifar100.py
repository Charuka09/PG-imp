"""
activation_ext_cifar100.py – Extract per-input activations entirely on GPU.

Uses pin_memory + non_blocking transfers and processes the full batch on GPU
before saving. Hooks capture GPU tensors and move to CPU only at save time.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from config import (TRAINED_MODELS_DIR, ACTIVATIONS_DIR, DATA_DIR,
                    CIFAR100_MEAN, CIFAR100_STD, NUM_CLASSES, LAYERS)
from models import Conv2Net, Conv6Net, VGG16CIFAR

device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PER_CLASS = 100
N_PER_CLASS   = 1000


def balanced_subset(dataset, n):
    buckets = {c: [] for c in range(len(dataset.classes))}
    for idx, (_, lbl) in enumerate(dataset):
        if len(buckets[lbl]) < n:
            buckets[lbl].append(idx)
        if all(len(v) == n for v in buckets.values()):
            break
    return Subset(dataset, [i for lst in buckets.values() for i in lst])


def save_activations(model, layer_names, inputs, labels, save_dir,
                     prefix, counters, max_per_class=MAX_PER_CLASS):
    if inputs.size(0) == 0:
        return counters
    os.makedirs(save_dir, exist_ok=True)

    # Keep activations on GPU until we know which samples to save
    acts  = {}
    hooks = []
    mods  = dict(model.named_modules())

    for name in layer_names:
        if name in mods:
            def _hook(_, __, out, n=name):
                acts[n] = out   # stays on GPU
            hooks.append(mods[name].register_forward_hook(_hook))

    inputs_gpu = inputs.to(device, non_blocking=True)
    labels_gpu = labels.to(device, non_blocking=True)

    with torch.no_grad():
        preds = model(inputs_gpu).argmax(1)

    correct_mask = (preds == labels_gpu)

    for i in range(inputs.size(0)):
        ok  = bool(correct_mask[i].item())
        key = "correct" if ok else "incorrect"
        if counters[key] >= max_per_class:
            continue

        # Only now move to CPU for saving
        torch.save(
            {"input":       inputs[i].cpu(),
             "activations": {n: a[i].cpu() for n, a in acts.items()},
             "correct":     ok},
            os.path.join(save_dir, f"{prefix}_input{i}.pt"),
            pickle_protocol=4,
        )
        counters[key] += 1
        if counters["correct"] >= max_per_class and counters["incorrect"] >= max_per_class:
            break

    for h in hooks:
        h.remove()
    acts.clear()
    return counters


def extract(model_name, model, layer_names, dataset):
    model.to(device).eval()
    save_base = os.path.join(ACTIVATIONS_DIR, model_name)
    print(f"\n{'='*55}\nExtracting: {model_name}\n{'='*55}")

    for class_id in range(NUM_CLASSES):
        counters = {"correct": 0, "incorrect": 0}
        indices  = [i for i, (_, lbl) in enumerate(dataset) if lbl == class_id]
        # pin_memory so GPU transfer is async
        loader   = DataLoader(Subset(dataset, indices), batch_size=128,
                              shuffle=False, pin_memory=True, num_workers=2)
        save_dir = os.path.join(save_base, f"class_{class_id}")

        for b_idx, (inputs, labels) in enumerate(
            tqdm(loader, desc=f"  class {class_id:3d}", leave=False)
        ):
            counters = save_activations(model, layer_names, inputs, labels,
                                        save_dir, prefix=f"batch{b_idx}",
                                        counters=counters)
            if counters["correct"] >= MAX_PER_CLASS and counters["incorrect"] >= MAX_PER_CLASS:
                break

        print(f"  class {class_id:3d}  correct={counters['correct']}  "
              f"incorrect={counters['incorrect']}")

    torch.cuda.empty_cache()


# ── Dataset ───────────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
full_train = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=val_tf)
subset     = balanced_subset(full_train, N_PER_CLASS)

MODELS = {
    # "conv2net": (Conv2Net(NUM_CLASSES),   LAYERS["conv2net"], "conv2net_best.pth"),
    # "conv6net": (Conv6Net(NUM_CLASSES),   LAYERS["conv6net"], "conv6net_best.pth"),
    "vgg16":    (VGG16CIFAR(NUM_CLASSES), LAYERS["vgg16"],    "vgg16_best.pth"),
}

for name, (model, layers, ckpt) in MODELS.items():
    path = os.path.join(TRAINED_MODELS_DIR, ckpt)
    if not os.path.exists(path):
        print(f"⚠  {path} not found — skipping {name}"); continue
    model.load_state_dict(torch.load(path, map_location=device))
    extract(name, model, layers, subset)

print("\n Activations saved to", ACTIVATIONS_DIR)
