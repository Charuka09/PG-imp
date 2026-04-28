
import argparse
import os
from pathlib import Path
import subprocess
import sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser(description="CIFAR-100 original pipeline runner (conv2net/conv6net/lenet300_100)")
    ap.add_argument("--step", type=str, required=True, choices=["train","extract","pg","cosine","all"])
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--activations_dir", type=str, default="./activations_cifar100")
    ap.add_argument("--pg_dir", type=str, default="./pg_multi_models_cifar100")
    ap.add_argument("--maps_dir", type=str, default="./w_fc_maps_dual_cifar100")
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--max_save_per_class", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--pg_samples", type=int, default=50)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--model", type=str, default="all", choices=["all","conv2net","conv6net","lenet"])

    args = ap.parse_args()

    # Expected ckpt names
    conv2_ckpt = str(Path(args.ckpt_dir) / "cifar100_conv2net_best.pth")
    conv6_ckpt = str(Path(args.ckpt_dir) / "cifar100_conv6net_best.pth")
    lenet_ckpt = str(Path(args.ckpt_dir) / "cifar100_lenet_best.pth")

    here = Path(__file__).resolve().parent

    if args.step in ("train","all"):
        models = ["conv2net","conv6net","lenet"] if args.model=="all" else [args.model]
        for m in models:
            out = str(Path(args.ckpt_dir) / f"cifar100_{m}_best.pth")
            run([sys.executable, str(here/"train_cifar100_models.py"),
                "--model", m, "--data_root", args.data_root, "--out_path", out])

    if args.step in ("extract","all"):
        run([sys.executable, str(here/"activation_extract_cifar100_original.py"),
             "--base_activation_dir", args.activations_dir,
             "--data_root", args.data_root,
             "--n_per_class", str(args.n_per_class),
             "--max_save_per_class", str(args.max_save_per_class),
             "--batch_size", str(args.batch_size),
             "--conv2net_ckpt", conv2_ckpt,
             "--conv6net_ckpt", conv6_ckpt,
             "--lenet_ckpt", lenet_ckpt])

    if args.step in ("pg","all"):
        run([sys.executable, str(here/"pg_ext_cifar100_original.py"),
             "--base_load_dir", args.activations_dir,
             "--output_base", args.pg_dir,
             "--num_classes", "100",
             "--max_samples", str(args.pg_samples)])

    if args.step in ("cosine","all"):
        run([sys.executable, str(here/"cosine_sim_ext_cifar100_original.py"),
             "--base_load_dir", args.activations_dir,
             "--save_dir", args.maps_dir,
             "--num_classes", "100"])

if __name__ == "__main__":
    main()
