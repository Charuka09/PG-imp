import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from models import Conv2Net, Conv6Net, LeNet300_100

def save_activations_per_input(model, layer_names, inputs, labels, save_dir, device='cpu',
                               prefix="sample", counters=None, max_per_class=100):
    """
    Saves full activations (original style): output tensors from chosen layers.
    Stops once max_per_class is reached for correct and incorrect.
    """
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
            raise KeyError(f"Layer '{name}' not found in model.named_modules().")
        hooks.append(named[name].register_forward_hook(get_activation(name)))

    with torch.no_grad():
        outputs = model(inputs.to(device))
        # Safety checks to prevent CIFAR10 ckpt misuse
        if outputs.dim() != 2 or outputs.size(1) != 100:
            raise RuntimeError(f"Expected logits [B,100] for CIFAR-100, got {tuple(outputs.shape)}. "
                               f"Wrong checkpoint/model num_classes?")
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

def select_n_per_class(dataset, n=1000, num_classes=100):
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < n:
            class_indices[label].append(idx)
        if all(len(lst) == n for lst in class_indices.values()):
            break
    selected_indices = [idx for lst in class_indices.values() for idx in lst]
    return Subset(dataset, selected_indices)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--base_activation_dir", default="./activations_cifar100")
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--max_save", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std =(0.2675, 0.2565, 0.2761))
    ])
    cifar_dataset = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=cifar_transform)
    subset_cifar = select_n_per_class(cifar_dataset, n=args.n_per_class, num_classes=100)

    if args.model == "conv2net":
        model = Conv2Net(num_classes=100).to(device)
        layers = ['conv1', 'conv2', 'pool', 'fc']
    elif args.model == "conv6net":
        model = Conv6Net(num_classes=100).to(device)
        layers = ['conv1','conv2','conv3','conv4','conv5','conv6','pool1','pool2','pool3','fc1','fc2']
    else:
        model = LeNet300_100(input_size=3*32*32, num_classes=100).to(device)
        layers = ['fc1','fc2']  # original setup (hidden layers)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    save_base = os.path.join(args.base_activation_dir, args.model)
    os.makedirs(save_base, exist_ok=True)

    for class_id in range(100):
        counters = {'correct': 0, 'incorrect': 0}

        class_indices = [i for i, (_, label) in enumerate(subset_cifar) if label == class_id]
        class_subset = Subset(subset_cifar, class_indices)
        class_loader = DataLoader(class_subset, batch_size=args.batch_size, shuffle=False)

        save_dir = os.path.join(save_base, f"class_{class_id}")
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx, (inputs, labels) in enumerate(tqdm(class_loader, desc=f"{args.model} | class {class_id}", leave=False)):
            counters = save_activations_per_input(
                model, layers, inputs, labels, save_dir,
                device=device, prefix=f"batch{batch_idx}", counters=counters, max_per_class=args.max_save
            )
            if counters['correct'] >= args.max_save and counters['incorrect'] >= args.max_save:
                break

        print(f"[{args.model}] class {class_id}: saved correct={counters['correct']}, incorrect={counters['incorrect']}")

if __name__ == "__main__":
    main()
