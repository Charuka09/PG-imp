"""
SVD-style structured pruning for CIFAR-100 models.

Scores each convolution filter by the nuclear norm of its singular values.
Lower-score filters are pruned exactly to the requested per-layer ratio.
"""

import torch

from benchmark_pruning_common import parse_args, run_method


def svd_filter_scores(model, model_name, conv_seq, train_loader, device,
                      score_batches):
    scores = {}
    del model_name, train_loader, score_batches
    with torch.no_grad():
        for conv_a, _, _ in conv_seq:
            conv = getattr(model, conv_a)
            weight = conv.weight.detach().to(device=device, dtype=torch.float32)
            layer_scores = []
            for filt in weight:
                mat = filt.reshape(filt.shape[0], -1)
                svals = torch.linalg.svdvals(mat)
                layer_scores.append(svals.sum())
            scores[conv_a] = torch.stack(layer_scores).cpu().numpy()
    return scores


if __name__ == "__main__":
    args = parse_args("svd")
    run_method("svd", svd_filter_scores, args)
