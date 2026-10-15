"""Check how well style codes are ordered, at the token level (prior only, no image generation).

For base codes N it compares the style token sequences of N and N+gap, and of N and a random code:
  - token match : fraction of positions holding the same token
  - pos cosine  : mean cosine of the codebook vectors at each position
  - pooled cos  : cosine of the mean codebook vector of each sequence (order-free)
For the ordered scheme it also reports how similar a section's chunks are at level gap 1, 2, ...,
how many tokens fall inside the boosted window of their level, and the prior's NLL of the
stitched sequences (higher than the original scheme means further from what the prior generates).
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoConfig
from transformers.generation.configuration_utils import GenerationConfig

from models.model import StyleGenerator
from models.style_code import OrderedStyleCode, add_style_code_args, original_style_ids
from models.utils import patched_from_model_config

GenerationConfig.from_model_config = classmethod(patched_from_model_config)


def load_prior(model_path):
    path = os.path.join(model_path, "prior")
    prior = StyleGenerator._from_config(AutoConfig.from_pretrained(f"{path}/config.json"))
    prior.load_state_dict(torch.load(f"{path}/prior.pth", map_location="cpu"))
    return prior.to("cuda", dtype=torch.bfloat16).eval()


def load_codebook(model_path):
    state = torch.load(os.path.join(model_path, "codebook", "model.pth"), map_location="cpu")
    weight = state["quantizer.codebooks.0.codebook.weight"].float()
    return torch.nn.functional.normalize(weight, dim=1).cuda()  # == NormalizedEmbedding.get_norm_weight()


def similarity(a, b, E):
    ea, eb = E[a.view(-1)], E[b.view(-1)]
    return {
        "token match": (a == b).float().mean().item(),
        "pos cosine": (ea * eb).sum(-1).mean().item(),
        "pooled cos": torch.nn.functional.cosine_similarity(ea.mean(0), eb.mean(0), dim=0).item(),
    }


@torch.no_grad()
def prior_nll(prior, ids, forward_kwargs):
    """Mean NLL (nats/token) of ids[2:] under the prior at temperature 1, without sampling filters.

    Position 0 is skipped because StyleGenerator.forward rescales the logits of position 0 only.
    """
    logits = prior(input_ids=ids, **forward_kwargs).logits[0, 1:-1].float()
    return torch.nn.functional.cross_entropy(logits, ids[0, 2:]).item()


def summarise(rows):
    keys = rows[0].keys()
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}


def pair_stats(ids_fn, base_codes, gaps, random_codes, E):
    stats = {}
    for gap in gaps:
        stats[f"N vs N+{gap}"] = summarise([similarity(ids_fn(n), ids_fn(n + gap), E) for n in base_codes])
    stats["N vs random"] = summarise([similarity(ids_fn(n), ids_fn(r), E) for n, r in zip(base_codes, random_codes)])
    return stats


def print_table(title, stats):
    print(f"\n{title}")
    print(f"  {'pair':<14}" + "".join(f"{k:>20}" for k in next(iter(stats.values()))))
    for name, s in stats.items():
        print(f"  {name:<14}" + "".join(f"{m:>13.3f} ± {sd:.3f}" for m, sd in s.values()))


def main(args):
    with open(f"{args.model_path}/freq.json") as f:
        code_freq = json.load(f)
    forward_kwargs = dict(code_freq=code_freq, code_freq_threshold=args.freq_threshold, k=args.freq_k)
    prior = load_prior(args.model_path)
    E = load_codebook(args.model_path)

    ordered = OrderedStyleCode(prior, code_freq, args.freq_threshold, args.freq_k, token_order=args.token_order,
                               num_sections=args.num_sections, num_levels=args.num_levels,
                               level_sigma=args.level_sigma, bias_strength=args.bias_strength,
                               noise_seed=args.noise_seed, use_anchor=not args.no_anchor)
    rng = np.random.default_rng(args.sweep_seed)
    base_codes = rng.integers(0, ordered.capacity - max(args.gaps), size=args.num_pairs).tolist()
    random_codes = rng.integers(0, ordered.capacity, size=args.num_pairs).tolist()
    print(f"ordered scheme: {args.num_sections} sections x {ordered.section_len} tokens, {args.num_levels} levels, "
          f"sigma {args.level_sigma}, strength {args.bias_strength}, anchor {not args.no_anchor}, "
          f"capacity {ordered.capacity:.3g}")

    results = {"config": vars(args), "base_codes": base_codes}

    # Chunk-level: similarity between levels of the same section, by level gap.
    for s in range(args.num_sections):
        for v in range(args.num_levels):
            ordered.chunk(s, v)
    by_gap = {}
    for s in range(args.num_sections):
        for u in range(args.num_levels):
            for v in range(u + 1, args.num_levels):
                by_gap.setdefault(v - u, []).append(similarity(ordered.chunk(s, u), ordered.chunk(s, v), E))
    results["level gap"] = {g: summarise(rows) for g, rows in sorted(by_gap.items())}
    print_table("ordered: chunks of the same section, by level gap", {f"gap {g}": s for g, s in results["level gap"].items()})

    # Steering: fraction of chunk tokens within +-sigma of their level's window centre.
    rank = torch.from_numpy(ordered.rank).cuda()
    n_ranked = (rank >= 0).sum().item()
    spacing = n_ranked / args.num_levels
    inside = []
    for (s, v), ids in ordered._chunks.items():
        r = rank[ids.view(-1)]
        inside.append(((r >= 0) & ((r - (v + 0.5) * spacing).abs() <= args.level_sigma * spacing)).float().mean().item())
    window_share = min(2 * args.level_sigma * spacing, n_ranked) / n_ranked
    results["inside window"] = {"chunks": float(np.mean(inside)), "uniform": window_share}
    print(f"\nordered: tokens within ±sigma of their level centre: {np.mean(inside):.3f} "
          f"(uniform sampling would give ~{window_share:.3f})")

    results["ordered"] = pair_stats(ordered, base_codes, args.gaps, random_codes, E)
    print_table("ordered: style codes", results["ordered"])
    nll_codes = base_codes + random_codes
    results["nll ordered"] = float(np.mean([prior_nll(prior, ordered(c), forward_kwargs) for c in nll_codes]))

    if args.baseline:
        cache = {}

        def original(code):
            if code not in cache:
                cache[code] = original_style_ids(prior, code, code_freq, args.freq_threshold, args.freq_k)
            return cache[code]

        results["original"] = pair_stats(original, base_codes, args.gaps, random_codes, E)
        print_table("original: style codes", results["original"])
        results["nll original"] = float(np.mean([prior_nll(prior, original(c), forward_kwargs) for c in nll_codes]))

    print(f"\nprior NLL (nats/token): ordered {results['nll ordered']:.3f}"
          + (f", original {results['nll original']:.3f}" if args.baseline else ""))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str, default="./pretrained_models")
    parser.add_argument("--freq_threshold", type=int, default=90000)
    parser.add_argument("--freq_k", type=float, default=0.0001)
    parser.add_argument("--num_pairs", type=int, default=20, help="number of base codes N")
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--sweep_seed", type=int, default=0, help="seed for picking the base and random codes")
    parser.add_argument("--baseline", action="store_true", help="also measure the original scheme (slower)")
    parser.add_argument("--out", type=str, default=None, help="optional JSON file for the results")
    add_style_code_args(parser)
    main(parser.parse_args())
