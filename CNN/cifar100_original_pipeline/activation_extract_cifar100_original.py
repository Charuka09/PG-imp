import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from models import Conv2Net, Conv6Net, LeNet300_100
from load_models import _extract_state_dict  # robust state_dict extraction


def save_activations_per_input(model, layer_names, inputs, labels, save_dir, device='cpu',
                               prefix="sample", counters=None, max_per_class=100):
    if counters is None:
        counters = {'correct': 0, 'incorrect': 0}

    if inputs.size(0) == 0:
        return counters

    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    activations = {}
    hooks = []

    def get_activation(name):
        def hook(_m, _inp, out):
            activations[name] = out.detach().cpu()
        return hook

    named = dict(model.named_modules())
    for name in layer_names:
        if name not in named:
            raise KeyError(f"Layer '{name}' not found. Sample keys: {list(named.keys())[:30]}")
        hooks.append(named[name].register_forward_hook(get_activation(name)))

    with torch.no_grad():
        outputs = model(inputs.to(device))
        preds = outputs.argmax(dim=1).cpu()
        correct_mask = (preds == labels).cpu()

    for i in range(inputs.size(0)):
        correct_i = bool(correct_mask[i].item())

        if correct_i and counters['correct'] >= max_per_class:
            continue
        if (not correct_i) and counters['incorrect'] >= max_per_class:
            continue

        data_to_save = {
            'input': inputs[i].cpu(),
            'activations': {name: activations[name][i] for name in layer_names},
            'correct': correct_i
        }
        save_path = os.path.join(save_dir, f"{prefix}_input{i}.pt")
        torch.save(data_to_save, save_path, pickle_protocol=4)

        counters['correct' if correct_i else 'incorrect'] += 1

        if counters['correct'] >= max_per_class and counters['incorrect'] >= max_per_class:
            break

    for h in hooks:
        h.remove()

    return counters


def select_n_per_class(dataset, n=500):
    num_classes = len(dataset.classes)
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < n:
            class_indices[label].append(idx)
        if all(len(lst) == n for lst in class_indices.values()):
            break
    selected_indices = [idx for lst in class_indices.values() for idx in lst]
    return Subset(dataset, selected_indices)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = _extract_state_dict(ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model


def run_extraction(
    model_name: str,
    ckpt_path: str,
    device=None,
    base_activation_dir="./activations",
    data_root="./data",
    n_per_class=500,
    max_save_per_class=100,
    batch_size=100,
    head_only=True,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(base_activation_dir, exist_ok=True)

    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])

    cifar_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=cifar_transform)

    if n_per_class > 500:
        raise ValueError("CIFAR-100 train has 500 images per class. Use n_per_class <= 500.")

    subset_cifar = select_n_per_class(cifar_dataset, n=n_per_class)

    # -------------------
    # Build ONE model + head-only layers
    # -------------------
    model_name = model_name.lower()
    if model_name == "conv2net":
        model = Conv2Net(num_classes=100).to(device)
        layers = ['fc'] if head_only else ['conv1','conv2','pool','fc']
    elif model_name == "conv6net":
        model = Conv6Net(num_classes=100).to(device)
        layers = ['fc1','fc2'] if head_only else ['conv1','conv2','conv3','conv4','conv5','conv6','pool1','pool2','pool3','fc1','fc2']
    elif model_name == "lenet":
        model = LeNet300_100(input_size=3*32*32, num_classes=100).to(device)
        layers = ['fc1','fc2','fc3'] if head_only else ['fc1','fc2','fc3']
    else:
        raise ValueError("model_name must be one of conv2net, conv6net, lenet")

    print(f"\nExtracting activations for {model_name.upper()} (CIFAR-100) | head_only={head_only}")
    model = load_checkpoint(model, ckpt_path, device=device)

    save_base = os.path.join(base_activation_dir, model_name)
    os.makedirs(save_base, exist_ok=True)

    for class_id in range(100):
        counters = {'correct': 0, 'incorrect': 0}

        class_indices = [i for i, (_, label) in enumerate(subset_cifar) if label == class_id]
        class_subset = Subset(subset_cifar, class_indices)
        class_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=False)

        save_dir = os.path.join(save_base, f"class_{class_id}")
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx, (inputs, labels) in enumerate(tqdm(class_loader, desc=f"{model_name} | class {class_id}", leave=False)):
            counters = save_activations_per_input(
                model, layers, inputs, labels, save_dir,
                device=device, prefix=f"batch{batch_idx}", counters=counters,
                max_per_class=max_save_per_class
            )
            if counters['correct'] >= max_save_per_class and counters['incorrect'] >= max_save_per_class:
                break

        print(f"[{model_name}] class {class_id}: saved correct={counters['correct']}, incorrect={counters['incorrect']}")

    print(f"\nDone. Activations saved under: {base_activation_dir}/{model_name}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_activation_dir", type=str, default="./activations")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--max_save_per_class", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--head_only", action="store_true", help="Extract only head layers (FC/logits).")
    args = ap.parse_args()

    run_extraction(
        model_name=args.model,
        ckpt_path=args.ckpt,
        device=args.device,
        base_activation_dir=args.base_activation_dir,
        data_root=args.data_root,
        n_per_class=args.n_per_class,
        max_save_per_class=args.max_save_per_class,
        batch_size=args.batch_size,
        head_only=args.head_only,
    )