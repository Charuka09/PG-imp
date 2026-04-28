#!/usr/bin/env bash
# Full fine-tuning pipeline (BERT)
# Usage: bash run_full.sh

set -e

OUT_DIR="/home/charuka09/Documents/postPhD/mindula/icml/Bert/agnews/new/"
SCRIPT="main.py"

echo "=== FULL FINETUNE PIPELINE ==="

python $SCRIPT   --stage finetune   --model_name bert-base-uncased   --out_dir $OUT_DIR   --epochs 2   --lr 2e-5   --batch_size 32

python $SCRIPT   --stage extract   --out_dir $OUT_DIR   --target_layers 0,5,11   --max_per_class 100   --batch_size 32

python $SCRIPT   --stage pg   --out_dir $OUT_DIR   --target_layers 0,5,11   --max_samples_pg 50   --sigma 0.1   --chunk_size 256

python $SCRIPT   --stage importance   --out_dir $OUT_DIR   --target_layers 0,5,11   --pca_k 4

echo "=== FULL PIPELINE DONE ==="
