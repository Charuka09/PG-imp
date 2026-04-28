import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from models import Conv2Net, Conv6Net, LeNet300_100
from load_models import load_conv2net_from_pth, load_conv6net_from_pth, load_lenet_from_pth

# -------------------
# Helper functions 
# -------------------


def save_activations_per_input(model, layer_names, inputs, labels, save_dir, device='cpu',
                               prefix="sample", counters=None, max_per_class=100):
    """
    Save activations, input, and correctness per input.
    Stops once max_per_class is reached for correct and incorrect.
    """
    if counters is None:
        counters = {'correct': 0, 'incorrect': 0}

    if inputs.size(0) == 0:
        return counters  # skip empty batch

    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook

    for name in layer_names:
        layer = dict(model.named_modules())[name]
        hooks.append(layer.register_forward_hook(get_activation(name)))

    with torch.no_grad():
        outputs = model(inputs.to(device))
        preds = outputs.argmax(dim=1).cpu()
        correct_mask = (preds == labels).cpu()

    for i in range(inputs.size(0)):
        correct_i = correct_mask[i].item()

        # Skip if we've reached max for this category
        if correct_i and counters['correct'] >= max_per_class:
            continue
        if not correct_i and counters['incorrect'] >= max_per_class:
            continue

        data_to_save = {
            'input': inputs[i].cpu(),
            'activations': {name: act[i] for name, act in activations.items()},
            'correct': correct_i
        }
        save_path = os.path.join(save_dir, f"{prefix}_input{i}.pt")
        torch.save(data_to_save, save_path, pickle_protocol=4)

        # Increment counters
        if correct_i:
            counters['correct'] += 1
        else:
            counters['incorrect'] += 1

        # Stop early if both reached
        if counters['correct'] >= max_per_class and counters['incorrect'] >= max_per_class:
            break

    for h in hooks:
        h.remove()

    return counters

def select_n_per_class(dataset, n=1000):
    """Select exactly n samples per class for balanced extraction."""
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < n:
            class_indices[label].append(idx)
        if all(len(lst) == n for lst in class_indices.values()):
            break
    selected_indices = [idx for lst in class_indices.values() for idx in lst]
    return Subset(dataset, selected_indices)


# -------------------
# Config
# -------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_activation_dir = "/home/charuka09/Documents/postPhD/mindula/icml/activations"
os.makedirs(base_activation_dir, exist_ok=True)


# -------------------
# CIFAR10 setup
# -------------------

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
subset_cifar = select_n_per_class(cifar_dataset, n=1000)


# -------------------
# Models + layers to extract
# -------------------

models_cifar = {
    "conv2net": (
        # load_conv2net_from_pth("/home/charuka09/Documents/postPhD/mindula/icml/models/conv2_best.pth", device=device),
        Conv2Net().to(device),
        ['conv1', 'conv2', 'pool', 'fc'],
        "/home/charuka09/Documents/postPhD/mindula/icml/models/conv2_best.pth"
    ),
    "conv6net": (
        # load_conv6net_from_pth("/home/charuka09/Documents/postPhD/mindula/icml/models/conv6_best.pth", device=device),
        Conv6Net().to(device),
        ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6',
         'pool1', 'pool2', 'pool3', 'fc1', 'fc2'],
        "/home/charuka09/Documents/postPhD/mindula/icml/models/conv6_best.pth"
    )
}

for model_name, (model, layers, path) in models_cifar.items():
    print(f"\n Extracting activations for {model_name.upper()}...")
    model.load_state_dict(torch.load(path, map_location=device))
    save_base = os.path.join(base_activation_dir, model_name)
    os.makedirs(save_base, exist_ok=True)

    for class_id in range(10):
        counters = {'correct': 0, 'incorrect': 0}

        class_indices = [i for i, (_, label) in enumerate(subset_cifar) if label == class_id]
        class_subset = Subset(subset_cifar, class_indices)
        class_loader = DataLoader(class_subset, batch_size=100, shuffle=False)

        save_dir = os.path.join(save_base, f"class_{class_id}")
        os.makedirs(save_dir, exist_ok=True)

        print(f" Class {class_id}...")
        for batch_idx, (inputs, labels) in enumerate(tqdm(class_loader, desc=f"{model_name} | class {class_id}", leave=False)):
            counters = save_activations_per_input(
                model, layers, inputs, labels, save_dir,
                device=device, prefix=f"batch{batch_idx}", counters=counters
            )
            if counters['correct'] >= 100 and counters['incorrect'] >= 100:
                break  # done with this class


# -------------------
# MNIST setup: LeNet
# -------------------

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
subset_mnist = select_n_per_class(mnist_dataset, n=1000)

lenet_model = LeNet300_100().to(device)
lenet_model.load_state_dict(torch.load(
    "/home/charuka09/Documents/postPhD/mindula/icml/models/lenet300_100_best.pth",
    map_location=device
))
lenet_layers = ['fc1', 'fc2']
lenet_save_base = os.path.join(base_activation_dir, "lenet")
os.makedirs(lenet_save_base, exist_ok=True)

print(f"\n Extracting activations for LENET...")
for class_id in range(10):
    counters = {'correct': 0, 'incorrect': 0}

    class_indices = [i for i, (_, label) in enumerate(subset_mnist) if label == class_id]
    class_subset = Subset(subset_mnist, class_indices)
    class_loader = DataLoader(class_subset, batch_size=100, shuffle=False)

    save_dir = os.path.join(lenet_save_base, f"class_{class_id}")
    os.makedirs(save_dir, exist_ok=True)

    print(f" Class {class_id}...")
    for batch_idx, (inputs, labels) in enumerate(tqdm(class_loader, desc=f"lenet | class {class_id}", leave=False)):
        counters = save_activations_per_input(
            lenet_model, lenet_layers, inputs, labels, save_dir,
            device=device, prefix=f"batch{batch_idx}", counters=counters
        )
        if counters['correct'] >= 100 and counters['incorrect'] >= 100:
            break  # done with this class
