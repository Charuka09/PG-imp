
CIFAR-100 ORIGINAL PIPELINE (Conv2Net / Conv6Net / LeNet300-100)

This package is a faithful CIFAR-100 adaptation of your original scripts:
- activation_ext3.py  -> activation_extract_cifar100_original.py
- pg_ext_2.py         -> pg_ext_cifar100_original.py
- cosine_sim_ext_2.py -> cosine_sim_ext_cifar100_original.py

LeNet300_100 is used as an MLP baseline on CIFAR-100 by flattening 3x32x32 => input_size=3072.

IMPORTANT
- CIFAR-100 train split has only 500 images per class; use --n_per_class <= 500.
- Saving full activations for 100 classes can be BIG on disk. If disk becomes an issue,
  reduce --max_save_per_class or edit layers saved (see script defaults).

1) Train all three models (creates checkpoints in ./checkpoints):
python main_cifar100_original.py --step train

2) Extract activations (CIFAR-100, per class, correct+incorrect):
python main_cifar100_original.py --step extract --activations_dir ./activations_cifar100

3) PG extraction:
python main_cifar100_original.py --step pg --activations_dir ./activations_cifar100 --pg_dir ./pg_multi_models_cifar100

4) Cosine similarity FC maps (optional):
python main_cifar100_original.py --step cosine --activations_dir ./activations_cifar100 --maps_dir ./w_fc_maps_dual_cifar100

5) Run everything:
python main_cifar100_original.py --step all

If you already have checkpoints, put them here:
./checkpoints/cifar100_conv2net_best.pth
./checkpoints/cifar100_conv6net_best.pth
./checkpoints/cifar100_lenet_best.pth
(or edit main_cifar100_original.py)