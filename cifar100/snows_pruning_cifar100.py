"""
SNOWS-style one-shot structured pruning for CIFAR-100 models.

This local benchmark uses one-shot activation-preservation scores: filters
with larger mean squared feature activations on a few training batches are
kept. Lower-score filters are pruned exactly to the requested per-layer ratio.
"""

import numpy as np
import torch

from benchmark_pruning_common import parse_args, run_method


def snows_filter_scores(model, model_name, conv_seq, train_loader, device,
                        score_batches):
    scores_sum = {conv_a: None for conv_a, _, _ in conv_seq}
    seen = 0
    hooks = []
    del model_name

    def make_hook(name):
        def hook(_, __, out):
            val = out.detach().float().pow(2).mean(dim=(0, 2, 3))
            if scores_sum[name] is None:
                scores_sum[name] = val
            else:
                scores_sum[name] = scores_sum[name] + val
        return hook

    model.eval()
    for conv_a, _, _ in conv_seq:
        hooks.append(getattr(model, conv_a).register_forward_hook(
            make_hook(conv_a)))

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx >= score_batches:
                break
            model(x.to(device, non_blocking=True))
            seen += 1

    for hook in hooks:
        hook.remove()

    if seen == 0:
        raise RuntimeError("SNOWS scoring saw no training batches.")

    scores = {}
    for conv_a, _, _ in conv_seq:
        if scores_sum[conv_a] is None:
            out_channels = getattr(model, conv_a).out_channels
            scores[conv_a] = np.ones(out_channels, dtype=np.float32)
        else:
            scores[conv_a] = (scores_sum[conv_a] / seen).cpu().numpy()
    return scores


if __name__ == "__main__":
    args = parse_args("snows")
    run_method("snows", snows_filter_scores, args)
