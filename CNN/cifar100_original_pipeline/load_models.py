# models.py
from __future__ import annotations

import math
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

StateDict = Dict[str, torch.Tensor]


# -------------------------
# Checkpoint/state utilities
# -------------------------
def _extract_state_dict(ckpt: Any) -> StateDict:
    """
    Supports:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}
    Strips leading "module." from DataParallel checkpoints.
    """
    state = ckpt
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not look like a state_dict.")

    out: StateDict = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


def _infer_adapt_hw(fc_in: int, out_channels: int) -> Tuple[int, int]:
    """
    Choose (H, W) such that out_channels * H * W == fc_in.
    Prefer square; otherwise pick a factor pair close to sqrt.
    """
    if fc_in % out_channels != 0:
        raise ValueError(
            f"Cannot infer spatial size: fc_in={fc_in} not divisible by out_channels={out_channels}"
        )
    hw = fc_in // out_channels
    s = int(math.isqrt(hw))
    if s * s == hw:
        return s, s
    for h in range(s, 0, -1):
        if hw % h == 0:
            return h, hw // h
    return 1, hw


# -------------------------
# CIFAR models
# -------------------------
class Conv2Net(nn.Module):
    """
    Flexible Conv2Net for CIFAR (3-channel input):
      conv1 -> relu -> conv2 -> relu -> pool -> adaptive -> flatten -> fc
    Named layers for hooks: conv1, conv2, pool, fc
    """
    def __init__(
        self,
        num_classes: int,
        c1: int,
        c2: int,
        k1: int,
        k2: int,
        fc_in: int,
        use_maxpool: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, kernel_size=k1, stride=1, padding=k1 // 2, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=k2, stride=1, padding=k2 // 2, bias=True)
        self.pool = nn.MaxPool2d(2, 2) if use_maxpool else nn.Identity()

        h, w = _infer_adapt_hw(fc_in=fc_in, out_channels=c2)
        self.adapt = nn.AdaptiveAvgPool2d((h, w))

        self.fc = nn.Linear(fc_in, num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Conv6Net(nn.Module):
    """
    Flexible Conv6Net for CIFAR (3-channel input):
      (conv1,conv2)->pool1 -> (conv3,conv4)->pool2 -> (conv5,conv6)->pool3
      -> adaptive -> flatten -> fc1 -> relu -> fc2
    Named layers for hooks:
      conv1..conv6, pool1,pool2,pool3, fc1,fc2
    """
    def __init__(
        self,
        num_classes: int,
        chans: Tuple[int, int, int, int, int, int],  # c1..c6
        ks: Tuple[int, int, int, int, int, int],     # k1..k6
        fc1_out: int,
        fc1_in: int,
        use_pools: Tuple[bool, bool, bool] = (True, True, True),
    ):
        super().__init__()
        c1, c2, c3, c4, c5, c6 = chans
        k1, k2, k3, k4, k5, k6 = ks

        self.conv1 = nn.Conv2d(3,  c1, kernel_size=k1, padding=k1 // 2, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=k2, padding=k2 // 2, bias=True)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=k3, padding=k3 // 2, bias=True)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=k4, padding=k4 // 2, bias=True)
        self.conv5 = nn.Conv2d(c4, c5, kernel_size=k5, padding=k5 // 2, bias=True)
        self.conv6 = nn.Conv2d(c5, c6, kernel_size=k6, padding=k6 // 2, bias=True)

        self.pool1 = nn.MaxPool2d(2, 2) if use_pools[0] else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2) if use_pools[1] else nn.Identity()
        self.pool3 = nn.MaxPool2d(2, 2) if use_pools[2] else nn.Identity()

        h, w = _infer_adapt_hw(fc_in=fc1_in, out_channels=c6)
        self.adapt = nn.AdaptiveAvgPool2d((h, w))

        self.fc1 = nn.Linear(fc1_in, fc1_out, bias=True)
        self.fc2 = nn.Linear(fc1_out, num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_conv2net_from_pth(
    path: str,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Conv2Net:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    state = _extract_state_dict(ckpt)

    # Expect keys: conv1.weight, conv2.weight, fc.weight (and biases)
    if "conv1.weight" not in state or "conv2.weight" not in state or "fc.weight" not in state:
        raise KeyError(f"Conv2Net keys not found. Sample keys: {list(state.keys())[:30]}")

    c1, in1, k1, _ = state["conv1.weight"].shape
    c2, in2, k2, _ = state["conv2.weight"].shape
    if in1 != 3:
        raise ValueError(f"Expected CIFAR input channels=3, got conv1.in={in1}")
    if in2 != c1:
        raise ValueError(f"conv2 expects in={in2} but conv1 out={c1}")

    fc_out, fc_in = state["fc.weight"].shape
    num_classes = fc_out  # trust checkpoint

    model = Conv2Net(num_classes=num_classes, c1=c1, c2=c2, k1=k1, k2=k2, fc_in=fc_in).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_conv6net_from_pth(
    path: str,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Conv6Net:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    state = _extract_state_dict(ckpt)

    needed = [f"conv{i}.weight" for i in range(1, 7)] + ["fc1.weight", "fc2.weight"]
    if not all(k in state for k in needed):
        raise KeyError(f"Conv6Net keys not found. Sample keys: {list(state.keys())[:40]}")

    c1, in1, k1, _ = state["conv1.weight"].shape
    c2, in2, k2, _ = state["conv2.weight"].shape
    c3, in3, k3, _ = state["conv3.weight"].shape
    c4, in4, k4, _ = state["conv4.weight"].shape
    c5, in5, k5, _ = state["conv5.weight"].shape
    c6, in6, k6, _ = state["conv6.weight"].shape

    if in1 != 3:
        raise ValueError(f"Expected CIFAR input channels=3, got conv1.in={in1}")

    fc1_out, fc1_in = state["fc1.weight"].shape
    fc2_out, fc2_in = state["fc2.weight"].shape
    num_classes = fc2_out  # trust checkpoint

    model = Conv6Net(
        num_classes=num_classes,
        chans=(c1, c2, c3, c4, c5, c6),
        ks=(k1, k2, k3, k4, k5, k6),
        fc1_out=fc1_out,
        fc1_in=fc1_in,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -------------------------
# MNIST model
# -------------------------
class LeNet300_100(nn.Module):
    """
    Classic LeNet-300-100 MLP for MNIST:
      784 -> 300 -> 100 -> num_classes
    Named layers for hooks: fc1, fc2, fc3
    """
    def __init__(self, num_classes: int = 10, in_features: int = 28 * 28):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_lenet_from_pth(
    path: str,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> LeNet300_100:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    state = _extract_state_dict(ckpt)

    # Expect keys: fc1.weight, fc2.weight, fc3.weight
    if "fc1.weight" not in state:
        raise KeyError(f"LeNet keys not found. Sample keys: {list(state.keys())[:30]}")

    fc1_out, in_features = state["fc1.weight"].shape
    if fc1_out != 300:
        # not fatal, but indicates a different MLP
        pass

    if "fc3.weight" in state:
        num_classes = state["fc3.weight"].shape[0]  # trust checkpoint

    model = LeNet300_100(num_classes=num_classes, in_features=in_features).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model