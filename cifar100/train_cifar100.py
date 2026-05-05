"""
train_cifar100.py – Train Conv2Net, Conv6Net, VGG16CIFAR on CIFAR-100.
Full GPU pipeline: pin_memory, non_blocking transfers, AMP mixed precision.
"""

import os, copy, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
import pandas as pd

from config import (TRAINED_MODELS_DIR, RESULTS_DIR, DATA_DIR,
                    CIFAR100_MEAN, CIFAR100_STD, NUM_CLASSES)
from models import Conv2Net, Conv6Net, VGG16CIFAR

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"          # Automatic Mixed Precision (FP16)
print(f"Device: {device}  |  AMP: {USE_AMP}")

# ── Transforms ────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# pin_memory + persistent_workers keep the GPU fed without CPU stalls
train_loader = DataLoader(
    datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf),
    batch_size=128, shuffle=True, num_workers=4,
    pin_memory=True, persistent_workers=True)
val_loader = DataLoader(
    datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf),
    batch_size=256, shuffle=False, num_workers=4,
    pin_memory=True, persistent_workers=True)


def train(model, optimizer, scheduler=None,
          num_epochs=200, patience=15, model_name="model"):
    criterion = nn.CrossEntropyLoss().to(device)
    scaler    = GradScaler(enabled=USE_AMP)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    records    = []
    t0         = time.time()

    for ep in range(num_epochs):
        model.train()
        run_loss, run_correct = 0.0, 0

        for x, y in train_loader:
            # non_blocking=True overlaps CPU→GPU transfer with compute
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            with autocast(enabled=USE_AMP):
                logits = model(x)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss    += loss.item() * x.size(0)
            run_correct += (logits.detach().argmax(1) == y).sum().item()

        t_loss = run_loss    / len(train_loader.dataset)
        t_acc  = run_correct / len(train_loader.dataset)

        # ── Validation (no AMP needed, just speed) ────────────
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y  = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    logits = model(x)
                    v_loss += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()

        v_loss = v_loss    / len(val_loader.dataset)
        v_acc  = v_correct / len(val_loader.dataset)

        records.append(dict(epoch=ep+1, train_loss=t_loss, train_acc=t_acc,
                            val_loss=v_loss, val_acc=v_acc))
        print(f"[{model_name}] {ep+1:3d}/{num_epochs}  "
              f"train {t_loss:.4f}/{t_acc:.4f}  val {v_loss:.4f}/{v_acc:.4f}")

        if v_loss < best_loss:
            best_loss  = v_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state,
                       os.path.join(TRAINED_MODELS_DIR, f"{model_name}_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {ep+1}")
                break

        if scheduler:
            scheduler.step(v_loss)

    model.load_state_dict(best_state)
    print(f"[{model_name}] Done in {(time.time()-t0)/60:.1f} min  "
          f"best val loss {best_loss:.4f}")
    pd.DataFrame(records).to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_metrics.csv"), index=False)


# ── Conv2Net ──────────────────────────────────────────────────
# m   = Conv2Net(NUM_CLASSES).to(device)
# opt = optim.Adam(m.parameters(), lr=2e-4, weight_decay=5e-4)
# sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.1)
# train(m, opt, sch, num_epochs=200, patience=15, model_name="conv2net")

# ── Conv6Net ──────────────────────────────────────────────────
# m   = Conv6Net(NUM_CLASSES).to(device)
# opt = optim.Adam(m.parameters(), lr=3e-4, weight_decay=5e-4)
# sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.1)
# train(m, opt, sch, num_epochs=200, patience=15, model_name="conv6net")

# ── VGG16 ─────────────────────────────────────────────────────
m   = VGG16CIFAR(NUM_CLASSES).to(device)
opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
# cosine schedule steps every epoch; pass None to train() and step manually
sch_cos = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
train(m, opt, scheduler=None, num_epochs=200, patience=20, model_name="vgg16")

print("\nAll models saved to", TRAINED_MODELS_DIR)
