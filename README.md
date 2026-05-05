# CIFAR-100 Head-Only Pipeline

This repo supports the CIFAR-100 head-only workflow for:

- `conv2net`
- `conv6net`
- `lenet`

VGG and other architectures are not part of this pipeline.

## Directory Layout

Run commands from `CNN/cifar100_original_pipeline`. The scripts still use the repo-root output directories:

```text
./data/
./checkpoints/
./activations/<model>/class_<k>/*.pt
./pg_head/<model>/<split>/pg1_data/*.npy
./pruning_results/
./pruning_results/pruned/
```

Model folder names are lowercase: `conv2net`, `conv6net`, `lenet`.

Older outputs may exist under `CNN/cifar100_original_pipeline/`. They are not used by the cleaned workflow unless you pass those paths explicitly.

If your shell does not have a `python` command, use `../../.venv/bin/python` from `CNN/cifar100_original_pipeline`.

## Head Layers

PG extraction uses only these layers:

```text
conv2net: fc
conv6net: fc1, fc2
lenet:    fc1, fc2, fc3
```

Activation `.pt` files store:

```python
{"activations": {layer: tensor_float16}, "correct": bool}
```

Raw input tensors are not saved.

## One-Model Commands

First enter the pipeline folder:

```bash
cd CNN/cifar100_original_pipeline
```

Train one model:

```bash
../../.venv/bin/python main_cifar100_original.py --step train --model conv6net
```

Extract head-only activations:

```bash
../../.venv/bin/python main_cifar100_original.py --step extract --model conv6net
```

Compute PG:

```bash
../../.venv/bin/python main_cifar100_original.py --step pg --model conv6net
```

Prune from PG:

```bash
../../.venv/bin/python main_cifar100_original.py --step prune --model conv6net \
  --pg_layer fc1
```

Run the full workflow for one model:

```bash
../../.venv/bin/python main_cifar100_original.py --step all --model conv6net
```

For `lenet`, pruning supports `--pg_layer fc1` or `--pg_layer fc2`.

For `conv2net`, head-only pruning exits with a clear error because its only head layer is logits (`fc`), and logits pruning is not meaningful for class-preserving CIFAR-100.

## Direct Script Commands

```bash
../../.venv/bin/python train_cifar100_models.py --model conv6net
```

```bash
python3 activation_extract_cifar100_original.py \
  --model conv6net \
  --ckpt checkpoints/cifar100_conv6net_best.pth
```

```bash
../../.venv/bin/python pg_ext_cifar100_head_only.py --model conv6net
```

```bash
python pg_pruning_cifar100_head_only.py \
  --model conv6net \
  --ckpt checkpoints/cifar100_conv6net_best.pth \
  --pg_layer fc1
```

## Quick Check

```bash
../../.venv/bin/python -c "import torch; from models import Conv6Net; m=Conv6Net(num_classes=100); print(m(torch.randn(2,3,32,32)).shape)"
```
