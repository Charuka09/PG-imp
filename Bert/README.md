# BERT PG-Based Pruning Pipeline (AG News)

This README documents the **end-to-end experimental pipeline** for adapting the CNN-based PG (Principal Graph) pruning framework to **BERT** on a text classification task.

The pipeline is implemented in `main.py` and supports **two training regimes**:

- **Full fine-tuning** (entire BERT encoder + classifier)
- **Head-only fine-tuning** (frozen encoder, train classifier only)

Each regime is treated as a **separate experimental run** with isolated checkpoints, activations, PG results, and pruning outputs.

---

## 1. Task and Model

- **Model**: `bert-base-uncased`
- **Dataset**: AG News (4 classes)
- **Objective**:
  - Separate samples into **correct vs incorrect predictions**
  - Extract FFN activations
  - Build functional connectivity graphs
  - Compute PG1 representations
  - Derive neuron importance
  - Perform structured FFN pruning and retraining

---

## 2. Directory Structure (per run)

Each run (full vs head-only) produces its own outputs:

```
Bert/agnews/
├── finetuned_model_full/
├── finetuned_model_head/
├── activations_full/
├── activations_head/
├── pg_full/
├── pg_head/
├── importance_ffn_full.npz
├── importance_ffn_head.npz
├── pruned_full_thr0.5/
├── pruned_head_thr0.5/
```

---

## 3. Five Pipeline Stages

Each experiment consists of **five executable stages**:

1. `finetune` – train the model
2. `extract` – save per-sample activations (correct vs incorrect)
3. `pg` – compute PG1 + affinity matrices
4. `importance` – compute PG-based neuron importance
5. `prune_retrain` – prune neurons and retrain

---

## 4. Executable Commands

Below are **all commands**, ready to copy–paste.

---

# A. Full Fine-Tuning (Baseline)

### 1. Fine-tune (full model)
```bash
python main.py --stage finetune --model_name bert-base-uncased --epochs 2 --lr 2e-5 --batch_size 32
```

### 2. Extract activations
```bash
python main.py --stage extract --target_layers 0,5,11 --max_per_class 100 --batch_size 32
```

### 3. Compute PG1 + sample affinity plots
```bash
python main.py --stage pg --target_layers 0,5,11 --max_samples_pg 50 --sigma 0.1 --chunk_size 256
```

### 4. Build PG-based neuron importance
```bash
python main.py --stage importance --target_layers 0,5,11 -pca_k 4
```

### 5. Prune + retrain
```bash
python main.py --stage prune_retrain --target_layers 0,5,11 --threshold 0.5 --epochs 2 --lr 2e-5

```

### 6. Threshold Sweep + CSV Logging
#### This stage runs multiple pruning thresholds in one go and logs results to a CSV file (crash‑safe), mirroring the CNN pruning experiments.

- For each threshold, the pipeline records:
  - Baseline accuracy & ECE
  - Post‑prune accuracy & ECE
  - Best retrained accuracy & ECE
  - Overall % of FFN neurons pruned
  - Per‑layer pruning percentages

- Results are appended incrementally to a CSV file, enabling long sweeps over SSH/tmux.

```bash
python main.py --stage sweep --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 2e-5 --batch_size 32
python main.py --stage sweep --prune_method random --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 2e-5 --batch_size 32
python main.py --stage sweep --prune_method l1 --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 2e-5 --batch_size 32
python main.py --stage sweep --prune_method l2 --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 2e-5 --batch_size 32
```

---

# B. Head-Only Fine-Tuning (Frozen Encoder)

### 1. Fine-tune (classifier only)
```bash
python main.py --stage finetune --model_name bert-base-uncased --freeze_encoder --epochs 3 --lr 5e-4 --batch_size 32
```

### 2. Extract activations
```bash
python main.py --stage extract --freeze_encoder --target_layers 0,5,11 --max_per_class 100 --batch_size 32
```

### 3. Compute PG1 + sample affinity plots
```bash
python main.py --stage pg --freeze_encoder --target_layers 0,5,11 --max_samples_pg 50 --sigma 0.1 --chunk_size 256
```

### 4. Build PG-based neuron importance
```bash
python main.py --stage importance --freeze_encoder --target_layers 0,5,11 --pca_k 4
```

### 5. Prune + retrain (head-only retraining)
```bash
python main.py --stage prune_retrain --freeze_encoder --target_layers 0,5,11 --threshold 0.5 --epochs 2 --lr 5e-4
```

### 6. Threshold Sweep + CSV Logging
#### This stage runs multiple pruning thresholds in one go and logs results to a CSV file (crash‑safe), mirroring the CNN pruning experiments.

- For each threshold, the pipeline records:
  - Baseline accuracy & ECE
  - Post‑prune accuracy & ECE
  - Best retrained accuracy & ECE
  - Overall % of FFN neurons pruned
  - Per‑layer pruning percentages

- Results are appended incrementally to a CSV file, enabling long sweeps over SSH/tmux.

```bash
python main.py --stage sweep --freeze_encoder --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 5e-4 --batch_size 32

python main.py --stage sweep --freeze_encoder --prune_method random --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 2e-5 --batch_size 32

python main.py --stage sweep --freeze_encoder --prune_method l1 --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 2e-5 --batch_size 32

python main.py --stage sweep --freeze_encoder --prune_method l2 --target_layers 0,5,11 --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --epochs 2 --lr 2e-5 --batch_size 32
```
---

## Notes

- BERT-base has **12 encoder layers (0–11)**.
- PG is computed on **FFN intermediate neurons (3072 units)** at the `[CLS]` token.
- Pruning is currently implemented via **structured masking** (zeroing neurons).

---

**Author**: Charuka Herath
