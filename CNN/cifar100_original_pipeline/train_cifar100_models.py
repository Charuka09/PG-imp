
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Conv2Net, Conv6Net, LeNet300_100

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(data_root, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
    ])
    train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def make_model(name):
    name = name.lower()
    if name == "conv2net":
        return Conv2Net(num_classes=100)
    if name == "conv6net":
        return Conv6Net(num_classes=100)
    if name in ("lenet", "lenet300_100"):
        return LeNet300_100(input_size=3*32*32, num_classes=100)
    raise ValueError(name)

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    tot=0; cor=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        tot += y.size(0)
        cor += (pred==y).sum().item()
    return 100.0*cor/max(tot,1)

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = make_model(args.model).to(device)
    train_loader, test_loader = get_loaders(args.data_root, args.batch_size)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.optim.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None

    best = -1.0
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        loss_sum=0.0; tot=0; cor=0
        for x,y in tqdm(train_loader, desc=f"train ep {ep}/{args.epochs}", leave=False):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*y.size(0)
            tot += y.size(0)
            cor += (logits.argmax(1)==y).sum().item()

        if scheduler: scheduler.step()
        train_acc = 100.0*cor/max(tot,1)
        test_acc = eval_acc(model, test_loader, device)
        print(f"[ep {ep:03d}] loss={loss_sum/max(tot,1):.4f} train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%")

        if test_acc > best:
            best = test_acc
            torch.save({"model_state_dict": model.state_dict(), "best_test_acc": best, "epoch": ep}, args.out_path)

    print(f"Saved best ckpt: {args.out_path} (best test acc {best:.2f}%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["conv2net","conv6net","lenet"])
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_path", type=str, default="./checkpoints/cifar100_{model}_best.pth")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--optim", type=str, default="sgd", choices=["sgd","adam"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()
    args.out_path = args.out_path.format(model=args.model)
    train(args)
