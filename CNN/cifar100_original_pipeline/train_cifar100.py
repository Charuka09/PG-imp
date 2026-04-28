#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import Conv2Net, Conv6Net, LeNet300_100

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cifar100_transforms(train=True):
    mean=(0.5071, 0.4867, 0.4408)
    std =(0.2675, 0.2565, 0.2761)
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def make_model(name: str, num_classes=100):
    name = name.lower()
    if name == "conv2net":
        return Conv2Net(num_classes=num_classes)
    if name == "conv6net":
        return Conv6Net(num_classes=num_classes)
    if name in ("lenet", "lenet300_100", "lenet300"):
        return LeNet300_100(input_size=3*32*32, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, cor = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        cor += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return 100.0 * cor / max(tot,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="./checkpoints/cifar100_{model}_best.pth")
    ap.add_argument("--opt", default="sgd", choices=["sgd","adam"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_ds = datasets.CIFAR100(args.data_root, train=True, download=True, transform=cifar100_transforms(True))
    test_ds  = datasets.CIFAR100(args.data_root, train=False, download=True, transform=cifar100_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = make_model(args.model, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.5*args.epochs), int(0.75*args.epochs), int(0.9*args.epochs)], gamma=0.2)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=max(args.lr, 1e-3), weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs//3), gamma=0.5)

    best = -1.0
    out_path = args.out.format(model=args.model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tot_loss, tot, cor = 0.0, 0, 0
        for x,y in tqdm(train_loader, desc=f"train {args.model} ep {ep}/{args.epochs}", leave=False):
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item()*y.size(0)
            tot += y.size(0)
            cor += (logits.argmax(1)==y).sum().item()

        sched.step()
        train_acc = 100.0*cor/max(tot,1)
        train_loss = tot_loss/max(tot,1)
        test_acc = evaluate(model, test_loader, device)
        print(f"[ep {ep:03d}] loss={train_loss:.4f} train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%")

        if test_acc > best:
            best = test_acc
            torch.save({"model_state_dict": model.state_dict(), "best_test_acc": best, "epoch": ep}, out_path)

    print(f"Saved best checkpoint: {out_path}  (best test acc {best:.2f}%)")

if __name__ == "__main__":
    main()
