#!/usr/bin/env bash
# Head-only fine-tuning pipeline (Frozen encoder)
# Usage: bash run_head.sh

set -e

OUT_DIR="/home/charuka09/Documents/postPhD/mindula/icml/Bert/agnews"
SCRIPT="main.py"

echo "=== HEAD-ONLY FINETUNE PIPELINE ==="

# python $SCRIPT   --stage finetune   --model_name bert-base-uncased   --out_dir $OUT_DIR   --freeze_encoder   --epochs 10   --lr 5e-4   --batch_size 32

# python $SCRIPT   --stage extract   --out_dir $OUT_DIR   --freeze_encoder   --target_layers 0,5,11   --max_per_class 100   --batch_size 32

# python $SCRIPT   --stage pg   --out_dir $OUT_DIR   --freeze_encoder   --target_layers 0,5,11   --max_samples_pg 50   --sigma 0.1   --chunk_size 256

# python $SCRIPT   --stage importance   --out_dir $OUT_DIR   --freeze_encoder   --target_layers 0,5,11   --pca_k 4

python $SCRIPT   --stage prune_retrain   --out_dir $OUT_DIR   --freeze_encoder   --target_layers 0,5,11   --threshold 0.5   --epochs 2   --lr 5e-4

# python main.py  --stage prune_retrain  --freeze_encoder   --target_layers 0,5,11   --threshold 0.5   --epochs 2   --lr 5e-4

echo "=== HEAD-ONLY PIPELINE DONE ==="
