"""
config.py – Central configuration for the CIFAR-100 PG-pruning project.
"""

import os

HERE               = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.join(HERE, "pg_project_output")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
ACTIVATIONS_DIR    = os.path.join(BASE_DIR, "activations")
AFFINITY_DIR       = os.path.join(BASE_DIR, "affinity_matrices")
PG_DIR             = os.path.join(BASE_DIR, "pg_data")
RESULTS_DIR        = os.path.join(BASE_DIR, "results")
DATA_DIR           = os.path.join(HERE, "data")

for _d in [TRAINED_MODELS_DIR, ACTIVATIONS_DIR, AFFINITY_DIR,
           PG_DIR, RESULTS_DIR, DATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset normalisation ─────────────────────────────────────
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

NUM_CLASSES = 100

# ── Layer names per model ─────────────────────────────────────
LAYERS = {
    "conv2net": ["conv1", "conv2", "pool", "fc"],
    "conv6net": ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6",
                 "pool1", "pool2", "pool3", "fc1", "fc2"],
    "vgg16":    ["conv1_1", "conv1_2", "pool1",
                 "conv2_1", "conv2_2", "pool2",
                 "conv3_1", "conv3_2", "conv3_3", "pool3",
                 "conv4_1", "conv4_2", "conv4_3", "pool4",
                 "conv5_1", "conv5_2", "conv5_3", "pool5",
                 "fc1", "fc2"],
}

# ── Conv layers only (for pruning) ────────────────────────────
CONV_ATTRS = {
    "conv2net": ["conv1", "conv2"],
    "conv6net": ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"],
    "vgg16":    ["conv1_1", "conv1_2",
                 "conv2_1", "conv2_2",
                 "conv3_1", "conv3_2", "conv3_3",
                 "conv4_1", "conv4_2", "conv4_3",
                 "conv5_1", "conv5_2", "conv5_3"],
}

BN_ATTRS = {
    "conv2net": ["bn1", "bn2"],
    "conv6net": ["bn1", "bn2", "bn3", "bn4", "bn5", "bn6"],
    "vgg16":    ["bn1_1", "bn1_2",
                 "bn2_1", "bn2_2",
                 "bn3_1", "bn3_2", "bn3_3",
                 "bn4_1", "bn4_2", "bn4_3",
                 "bn5_1", "bn5_2", "bn5_3"],
}

# ── FC layer used for PG importance ──────────────────────────
PG_FC_LAYER = {
    "conv2net": "fc",
    "conv6net": "fc2",
    "vgg16":    "fc1",
}
