# CIFAR-100 PG-Pruning — GPU Edition

Conv2Net, Conv6Net, VGG16CIFAR on CIFAR-100 with full GPU acceleration.

---

## Files

```
├── config.py                  # All paths and constants (edit BASE_DIR here)
├── models.py                  # Conv2Net, Conv6Net, VGG16CIFAR
├── pruning_utils.py           # Shared helpers (prune_conv, rebuild_fc, …)
├── train_cifar100.py          # Step 1
├── activation_ext_cifar100.py # Step 2
├── cosine_sim_ext_cifar100.py # Step 3
├── pg_ext_cifar100.py         # Step 4
├── hrank_cifar100.py          # Step 5  – HRank for all three models
├── pg_pruning_cifar100.py     # Step 6  – PG pruning for all three models
└── requirements.txt
```

---

## What runs on GPU

| Script | GPU ops |
|---|---|
| `train_cifar100.py` | Forward/backward pass, **AMP FP16**, pin_memory loaders |
| `activation_ext_cifar100.py` | Model forward pass, activations stay on GPU until save |
| `cosine_sim_ext_cifar100.py` | Full cosine matrix via `F.normalize + matmul` on CUDA |
| `pg_ext_cifar100.py` | Cosine affinity **and** eigendecomposition (`torch.linalg.eigh`) on CUDA |
| `hrank_cifar100.py` | Feature-map SVD on GPU, FC fine-tune with **AMP FP16** |
| `pg_pruning_cifar100.py` | Importance matmul on GPU, FC fine-tune with **AMP FP16** |

CPU-only (by design):
- PCA (`sklearn`) — input is at most (100 × 512), GPU overhead not worthwhile.
- Saving `.pt` / `.npy` files — I/O bound.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Pipeline

```bash
python train_cifar100.py          # Step 1 – train
python activation_ext_cifar100.py # Step 2 – extract activations
python cosine_sim_ext_cifar100.py # Step 3 – affinity matrices
python pg_ext_cifar100.py         # Step 4 – PG1 scores
python hrank_cifar100.py          # Step 5 – HRank pruning
python pg_pruning_cifar100.py     # Step 6 – PG pruning
```

## Running baselines

```bash
cd /home/lunet/llckbhk/Documents/PG-imp
.venv/bin/python cifar100/svd_pruning_cifar100.py --models conv2net conv6net vgg16 --ratios 0.7
.venv/bin/python cifar100/sliming_pruning_cifar100.py --models conv2net conv6net vgg16 --ratios 0.7
.venv/bin/python cifar100/snows_pruning_cifar100.py --models conv2net conv6net vgg16 --ratios 0.7

```

Output root: `./pg_project_output/` — change `BASE_DIR` in `config.py`.

---

## Key GPU implementation notes

### AMP (Automatic Mixed Precision)
`torch.cuda.amp.autocast` + `GradScaler` are used in both training and
FC fine-tuning. This halves memory bandwidth and gives ~1.5–2× throughput
on Ampere/Turing GPUs with no accuracy loss.

### Cosine affinity on GPU
```python
t   = F.normalize(X_gpu, dim=1)   # unit-norm rows
mat = t @ t.T                      # (N, N) or (units, units)
```
One kernel call replaces the nested chunked loop.

### Eigendecomposition on GPU
`torch.linalg.eigh` (CUSOLVER backend) replaces `scipy.linalg.eigh`.
Run in `float64` for numerical stability. The second-largest eigenpair
is `eigvecs[:, -2]` (eigh returns ascending order).

### No joblib parallelism in pg_ext
CUDA contexts cannot be safely forked. Sequential GPU iterations are
faster than parallel CPU workers for this workload anyway.

### pin_memory + non_blocking
All DataLoaders use `pin_memory=True`; all `.to(device)` calls use
`non_blocking=True`, enabling CPU→GPU transfer to overlap with compute.

### zero_grad(set_to_none=True)
Avoids zeroing gradient tensors; instead sets them to `None`, saving one
memset per parameter per step.

---

## Output

```
pg_project_output/
├── trained_models/          conv2net_best.pth  conv6net_best.pth  vgg16_best.pth
├── activations/             <model>/class_<id>/*.pt
├── affinity_matrices/       <model>/class_<id>_{correct,incorrect}.npy
├── pg_data/                 <model>/{correct,incorrect}/pg1_data/*.npy
└── results/
    ├── *_metrics.csv                 training curves
    ├── hrank_conv2net_cifar100.csv
    ├── hrank_conv6net_cifar100.csv
    ├── hrank_vgg16_cifar100.csv
    ├── pg_conv2net_cifar100.csv
    ├── pg_conv6net_cifar100.csv
    └── pg_vgg16_cifar100.csv
```

Each pruning CSV records: threshold/ratio, kept filters per layer, pruning %,
pre/post accuracy, FLOPs, params, ECE, epochs to convergence.

---

## VGG16CIFAR architecture

Standard VGG-16 with BatchNorm, adapted for 32×32 input.
Five 2×2 max-pool stages: **32→16→8→4→2→1**.
After pool5: 512×1×1 → flatten → 512-dim FC → Dropout → 100-class output.
All 13 conv layers are named individually (`conv1_1`…`conv5_3`) so activation
hooks and pruning work with no special casing.

## PG importance

The FC layer used for PG1 contrastive signals:

| Model | FC layer | Dim |
|---|---|---|
| Conv2Net | `fc` | 16 384 → 100 |
| Conv6Net | `fc2` | 256 → 100 |
| VGG16 | `fc1` | 512 → 512 |

VGG16 uses `fc1` (not `fc2`) because 512 dimensions provide richer PCA
structure than the 100-dim output layer across 100 classes.