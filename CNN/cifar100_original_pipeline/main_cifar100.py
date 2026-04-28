#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main runner (original-style) for CIFAR-100 with:
  conv2net, conv6net, lenet300_100

Steps:
  train    -> train_cifar100.py
  extract  -> activation_ext_cifar100.py
  pg       -> pg_ext_cifar100.py
  prune    -> pg_pruning_cifar100.py
  cosine   -> cosine_sim_ext_cifar100.py

Examples:
  python main_cifar100.py train --model conv2net --out ./checkpoints/cifar100_conv2net_best.pth
  python main_cifar100.py extract --model conv2net --ckpt ./checkpoints/cifar100_conv2net_best.pth --act_dir ./activations_cifar100
  python main_cifar100.py pg --model conv2net --act_dir ./activations_cifar100 --pg_dir ./pg_cifar100
  python main_cifar100.py prune --model conv2net --ckpt ./checkpoints/cifar100_conv2net_best.pth --pg_dir ./pg_cifar100 --thresholds 0.3 0.5 0.7
"""

import argparse, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)

    # train
    p = sub.add_parser("train")
    p.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--device", default="")
    p.add_argument("--out", default="./checkpoints/cifar100_{model}_best.pth")
    p.add_argument("--opt", default="sgd", choices=["sgd","adam"])

    # extract
    p = sub.add_parser("extract")
    p.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--act_dir", default="./activations_cifar100")
    p.add_argument("--n_per_class", type=int, default=1000)
    p.add_argument("--max_save", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--device", default="")

    # pg
    p = sub.add_parser("pg")
    p.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    p.add_argument("--act_dir", default="./activations_cifar100")
    p.add_argument("--pg_dir", default="./pg_cifar100")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--n_jobs", type=int, default=-1)

    # prune
    p = sub.add_parser("prune")
    p.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pg_dir", default="./pg_cifar100")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs_retrain", type=int, default=80)
    p.add_argument("--lr_retrain", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--pca_n_components", type=int, default=4)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    p.add_argument("--out_dir", default="./pruned_checkpoints_cifar100")
    p.add_argument("--pg_layer", default="")

    # cosine
    p = sub.add_parser("cosine")
    p.add_argument("--model", required=True, choices=["conv2net","conv6net","lenet"])
    p.add_argument("--act_dir", default="./activations_cifar100")
    p.add_argument("--save_dir", default="./w_fc_maps_dual_cifar100")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--chunk_size", type=int, default=256)

    args = ap.parse_args()
    step = args.step

    if step == "train":
        cmd = [sys.executable, os.path.join(HERE, "train_cifar100.py"),
               "--model", args.model,
               "--data_root", args.data_root,
               "--epochs", str(args.epochs),
               "--batch_size", str(args.batch_size),
               "--lr", str(args.lr),
               "--weight_decay", str(args.weight_decay),
               "--out", args.out,
               "--opt", args.opt]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd)

    elif step == "extract":
        cmd = [sys.executable, os.path.join(HERE, "activation_ext_cifar100.py"),
               "--model", args.model,
               "--ckpt", args.ckpt,
               "--data_root", args.data_root,
               "--base_activation_dir", args.act_dir,
               "--n_per_class", str(args.n_per_class),
               "--max_save", str(args.max_save),
               "--batch_size", str(args.batch_size)]
        if args.device:
            cmd += ["--device", args.device]
        run(cmd)

    elif step == "pg":
        cmd = [sys.executable, os.path.join(HERE, "pg_ext_cifar100.py"),
               "--model", args.model,
               "--base_load_dir", args.act_dir,
               "--output_base", args.pg_dir,
               "--sigma", str(args.sigma),
               "--alpha", str(args.alpha),
               "--t", str(args.t),
               "--chunk_size", str(args.chunk_size),
               "--n_jobs", str(args.n_jobs)]
        run(cmd)

    elif step == "prune":
        cmd = [sys.executable, os.path.join(HERE, "pg_pruning_cifar100.py"),
               "--model", args.model,
               "--ckpt", args.ckpt,
               "--pg_root", args.pg_dir,
               "--data_root", args.data_root,
               "--batch_size", str(args.batch_size),
               "--epochs_retrain", str(args.epochs_retrain),
               "--lr_retrain", str(args.lr_retrain),
               "--patience", str(args.patience),
               "--pca_n_components", str(args.pca_n_components),
               "--out_dir", args.out_dir]
        if args.device:
            cmd += ["--device", args.device]
        if args.pg_layer:
            cmd += ["--pg_layer", args.pg_layer]
        # thresholds
        cmd += ["--thresholds"] + [str(x) for x in args.thresholds]
        run(cmd)

    elif step == "cosine":
        cmd = [sys.executable, os.path.join(HERE, "cosine_sim_ext_cifar100.py"),
               "--model", args.model,
               "--base_load_dir", args.act_dir,
               "--save_dir", args.save_dir,
               "--max_samples", str(args.max_samples),
               "--chunk_size", str(args.chunk_size)]
        run(cmd)

if __name__ == "__main__":
    main()
