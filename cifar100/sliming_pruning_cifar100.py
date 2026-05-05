"""
SLIMING-style structured pruning for CIFAR-100 models.

This local benchmark uses a singular-value slimming proxy: each filter's
singular-value energy is weighted by the absolute BatchNorm scale gamma.
Lower-score filters are pruned exactly to the requested per-layer ratio.
"""

import torch

from benchmark_pruning_common import parse_args, run_method


def sliming_filter_scores(model, model_name, conv_seq, train_loader, device,
                          score_batches):
    scores = {}
    del model_name, train_loader, score_batches
    with torch.no_grad():
        for conv_a, bn_a, _ in conv_seq:
            conv = getattr(model, conv_a)
            bn = getattr(model, bn_a)
            weight = conv.weight.detach().to(device=device, dtype=torch.float32)
            gamma = bn.weight.detach().abs().to(device=device, dtype=torch.float32)

            layer_scores = []
            for idx, filt in enumerate(weight):
                mat = filt.reshape(filt.shape[0], -1)
                svals = torch.linalg.svdvals(mat)
                layer_scores.append(svals.sum() * gamma[idx])
            scores[conv_a] = torch.stack(layer_scores).cpu().numpy()
    return scores


if __name__ == "__main__":
    args = parse_args("sliming")
    run_method("sliming", sliming_filter_scores, args)
