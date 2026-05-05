"""
models.py – Conv2Net, Conv6Net, VGG16CIFAR for CIFAR-100 (32×32, 100 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2Net(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, 3, padding=1);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.pool    = nn.MaxPool2d(2)
        self.fc      = nn.Linear(64 * 16 * 16, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Conv6Net(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  32,  3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32,  64,  3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,  64,  3, padding=1); self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(64,  128, 3, padding=1); self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1); self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.fc1     = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class VGG16CIFAR(nn.Module):
    """
    VGG-16 with BatchNorm adapted for 32×32 input.

    Spatial sizes:  32 → 16 → 8 → 4 → 2 → 1  (five 2×2 max-pools)
    After pool5: 512×1×1 → flatten → 512.
    Classifier: Linear(512,512) → Dropout → Linear(512, num_classes).

    All conv/bn layers are named individually (conv1_1, bn1_1, …) so
    hook-based activation extraction and structured pruning work unchanged.
    """

    def __init__(self, num_classes=100):
        super().__init__()
        # Block 1  (32→16)
        self.conv1_1 = nn.Conv2d(3,   64,  3, padding=1); self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64,  64,  3, padding=1); self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1   = nn.MaxPool2d(2)
        # Block 2  (16→8)
        self.conv2_1 = nn.Conv2d(64,  128, 3, padding=1); self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1); self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2   = nn.MaxPool2d(2)
        # Block 3  (8→4)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1); self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1); self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1); self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3   = nn.MaxPool2d(2)
        # Block 4  (4→2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1); self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1); self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1); self.bn4_3 = nn.BatchNorm2d(512)
        self.pool4   = nn.MaxPool2d(2)
        # Block 5  (2→1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1); self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1); self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1); self.bn5_3 = nn.BatchNorm2d(512)
        self.pool5   = nn.MaxPool2d(2)
        # Classifier
        self.fc1     = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool4(x)
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
