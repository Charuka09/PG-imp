import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ======================================================
# Conv2Net – for Cifar10
# ======================================================
class Conv2Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)  # match conv1 out_channels

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)  # match conv2 out_channels

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Fully connected layer
        # CIFAR10 input 32x32 → after pool 16x16 → conv2 output 64 channels
        self.fc = nn.Linear(64 * 16 * 16, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ======================================================
# Conv6Net – for CIFAR10
# ======================================================
class Conv6Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv blocks
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 32x32 → 16x16

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 16x16 → 8x8

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)  # 8x8 → 4x4

        # FC
        self.fc1 = nn.Linear(128*4*4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

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
        x = self.fc2(x)
        return x

# ======================================================
# LeNet-300-100 – for MNIST
# ======================================================

class LeNet300_100(nn.Module): 
    def __init__(self, input_size=28*28, num_classes=10):
        super(LeNet300_100, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x


# ======================================================
# CIFAR-adapted ResNet (ResNet32)
# ======================================================
# Based on standard ResNet implementation for CIFAR (3x32x32 input)
from torchvision.models.resnet import BasicBlock, ResNet

class ResNetCIFAR(ResNet):
    def __init__(self, num_classes=100, depth=32):
        # CIFAR-ResNet uses smaller depth (e.g., ResNet32 has n=5 blocks per stage)
        block = BasicBlock
        layers = [(depth - 2) // 6] * 3  # number of blocks per stage
        super().__init__(block, layers, num_classes=num_classes)
        # override first conv and maxpool for 32x32 images
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # no maxpool for small images
        self.fc = nn.Linear(16 * block.expansion, num_classes)


# ======================================================
# VGG-16 for CIFAR-100 (32x32)
# - Explicit blocks/pools so you can hook: pool1..pool5, fc1..fc3
# ======================================================
class VGG16CIFAR(nn.Module):
    def __init__(self, num_classes: int = 100, use_bn: bool = True, dropout: float = 0.5):
        super().__init__()
        def conv3x3(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # VGG16 blocks: (2,2,3,3,3) convs with MaxPool between blocks
        self.block1 = nn.Sequential(conv3x3(3, 64), conv3x3(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)   # 32 -> 16

        self.block2 = nn.Sequential(conv3x3(64, 128), conv3x3(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)   # 16 -> 8

        self.block3 = nn.Sequential(conv3x3(128, 256), conv3x3(256, 256), conv3x3(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)   # 8 -> 4

        self.block4 = nn.Sequential(conv3x3(256, 512), conv3x3(512, 512), conv3x3(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2)   # 4 -> 2

        self.block5 = nn.Sequential(conv3x3(512, 512), conv3x3(512, 512), conv3x3(512, 512))
        self.pool5 = nn.MaxPool2d(2, 2)   # 2 -> 1

        # CIFAR ends at 1x1 spatial; flatten = 512
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Optional: keep a 'classifier' Sequential for familiarity
        self.classifier = nn.Sequential(
            self.fc1, nn.ReLU(True), self.dropout,
            self.fc2, nn.ReLU(True), self.dropout,
            self.fc3
        )

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.block5(x))
        x = torch.flatten(x, 1)  # [B,512]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x