"""CSD style evaluation of style-code schemes (original vs ordered).

Metrics, all from CSD style embeddings (Somepalli et al., 2024, "Measuring Style Similarity in
Diffusion Models"), as in the CoTyle paper plus a locality metric:
  - consistency : mean CSD similarity between images of the same code (different prompts)
  - diversity   : 1 - mean CSD similarity between images of different codes (same prompt)
  - locality    : mean CSD similarity between images of codes N and N+gap (same prompt and seed),
                  next to the similarity of random code pairs

Usage:
  python style_csd_eval.py generate --model_path <CoTyle dir> --out outputs/csd_eval
  python style_csd_eval.py score --out outputs/csd_eval

`generate` runs inference_batch.py once per scheme and image set; arguments it does not know are
passed on to inference_batch.py (e.g. --bias_strength 4). Images that already exist are not
regenerated, so an interrupted `generate` can be restarted with the same command.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
CSD_REPO = "tomg-group-umd/CSD-ViT-L"


def image_path(out, scheme, image_set, code, prompt_i):
    # inference_batch.py saves {output_path}/batch/{style_code}-{prompt_i}.png
    return os.path.join(out, scheme, image_set, "batch", f"{code}-{prompt_i}.png")


def generate(args, passthrough):
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.code_seed)
    # Ordered codes are taken modulo 10**7 (7 sections x 10 levels); stay below it so N + gap does not wrap.
    codes = sorted(rng.choice(10**7 - max(args.gaps), size=args.num_codes, replace=False).tolist())
    bases = codes[:args.num_locality]
    neighbours = sorted({b + g for b in bases for g in args.gaps} - set(codes))

    with open(args.prompt_file) as f:
        prompts = [line.strip() for line in f if line.strip()][:args.num_prompts]
    prompt_files = {"main": os.path.join(args.out, "prompts_main.txt"),
                    "locality": os.path.join(args.out, "prompts_locality.txt")}
    with open(prompt_files["main"], "w") as f:
        f.write("\n".join(prompts))
    with open(prompt_files["locality"], "w") as f:
        f.write(prompts[0])  # same prompt index 0 and seed as the main set, so N and N+gap are comparable

    manifest = dict(schemes=args.schemes, codes=codes, locality_bases=bases, gaps=args.gaps,
                    prompts=prompts, seed=args.seed, accelerate=not args.no_accelerate,
                    inference_batch_args=passthrough)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    runs = [("main", codes, len(prompts)), ("locality", neighbours, 1)]
    for scheme in args.schemes:
        for image_set, run_codes, n_prompts in runs:
            expected = [image_path(args.out, scheme, image_set, c, p) for c in run_codes for p in range(n_prompts)]
            if all(os.path.exists(p) for p in expected):
                print(f"[skip] {scheme}/{image_set}: all {len(expected)} images exist")
                continue
            cmd = [sys.executable, os.path.join(REPO, "inference_batch.py"),
                   "--model_path", args.model_path,
                   "--style_mode", scheme,
                   "--style_code", *map(str, run_codes),
                   "--prompt_file_path", prompt_files[image_set],
                   "--output_path", os.path.join(args.out, scheme, image_set),
                   "--seed", str(args.seed),
                   "--batch_size", str(args.batch_size),
                   "--skip_existing",
                   *([] if args.no_accelerate else ["--accelerate"]),
                   *passthrough]
            done = sum(os.path.exists(p) for p in expected)
            print(f"[run] {scheme}/{image_set}: {len(expected) - done} of {len(expected)} images to generate", flush=True)
            subprocess.run(cmd, cwd=REPO, check=True)


class CSDStyleEncoder:
    """CSD style embedding: CLIP ViT-L/14 visual backbone (proj removed) followed by the style projection.

    Mirrors CSD_CLIP and transforms_branch0 in github.com/learn2phoenix/CSD; the embedding is the
    L2-normalised `style_output`.
    """

    def __init__(self, device="cuda"):
        import open_clip
        import torch
        import torchvision.transforms as T
        from huggingface_hub import hf_hub_download

        state = torch.load(hf_hub_download(CSD_REPO, "pytorch_model.bin"), map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}

        # OpenAI CLIP ViT-L/14 uses QuickGELU; the weights come from the CSD checkpoint, not from CLIP.
        self.backbone = open_clip.create_model("ViT-L-14-quickgelu", pretrained=None).visual
        backbone_state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if set(missing) - {"proj"} or unexpected:
            raise RuntimeError(f"CSD backbone mismatch: missing {missing}, unexpected {unexpected}")
        self.backbone.proj = None
        self.backbone.to(device).eval()
        self.style_proj = state["last_layer_style"].float().to(device)

        self.device = device
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, paths, batch_size=16):
        import torch
        from PIL import Image

        out = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                x = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
                feats = self.backbone(x.to(self.device)) @ self.style_proj
                out.append(torch.nn.functional.normalize(feats, dim=1).cpu())
        return torch.cat(out).numpy()


def mean_sd(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values)), "n": len(values)}


def score(args):
    with open(os.path.join(args.out, "manifest.json")) as f:
        m = json.load(f)
    codes, bases, gaps, n_prompts = m["codes"], m["locality_bases"], m["gaps"], len(m["prompts"])
    encoder = CSDStyleEncoder()

    results = {}
    for scheme in m["schemes"]:
        keys = [(c, p) for c in codes for p in range(n_prompts)]
        paths = [image_path(args.out, scheme, "main", c, p) for c, p in keys]
        loc_keys = sorted({(b + g, 0) for b in bases for g in gaps} - set(keys))
        paths += [image_path(args.out, scheme, "locality", c, p) for c, p in loc_keys]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"[{scheme}] skipped: {len(missing)} images missing, e.g. {missing[0]}")
            continue
        emb = dict(zip(keys + loc_keys, encoder(paths)))
        sim = lambda a, b: float(emb[a] @ emb[b])
        # Per-image embeddings, for analyses beyond the summary metrics (row i is image key i).
        np.savez(os.path.join(args.out, f"csd_embeddings_{scheme}.npz"),
                 keys=np.array(keys + loc_keys), embeddings=np.stack([emb[k] for k in keys + loc_keys]))

        consistency = [np.mean([sim((c, p), (c, q)) for p, q in itertools.combinations(range(n_prompts), 2)])
                       for c in codes]
        cross = {p: [sim((a, p), (b, p)) for a, b in itertools.combinations(codes, 2)] for p in range(n_prompts)}
        r = {"consistency": mean_sd(consistency),
             "diversity": mean_sd([1 - s for p in cross for s in cross[p]]),
             "locality": {f"N vs N+{g}": mean_sd([sim((b, 0), (b + g, 0)) for b in bases]) for g in gaps}}
        r["locality"]["N vs random"] = mean_sd(cross[0])
        results[scheme] = r

    print(f"\nCSD style similarity ({'accelerate' if m['accelerate'] else '40-step'} mode, "
          f"{len(codes)} codes x {n_prompts} prompts, locality on {len(bases)} base codes)")
    for scheme, r in results.items():
        print(f"\n[{scheme}]")
        print(f"  consistency (same code)      {r['consistency']['mean']:.4f} ± {r['consistency']['sd']:.4f}")
        print(f"  diversity (1 - cross-code)   {r['diversity']['mean']:.4f} ± {r['diversity']['sd']:.4f}")
        for name, s in r["locality"].items():
            print(f"  locality {name:<14}      {s['mean']:.4f} ± {s['sd']:.4f}  (n={s['n']})")
    with open(os.path.join(args.out, "csd_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {os.path.join(args.out, 'csd_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate the evaluation images with inference_batch.py")
    gen.add_argument("--model_path", required=True)
    gen.add_argument("--out", default="outputs/csd_eval")
    gen.add_argument("--schemes", nargs="+", default=["original", "ordered"], choices=["original", "ordered"])
    gen.add_argument("--num_codes", type=int, default=20)
    gen.add_argument("--num_prompts", type=int, default=4)
    gen.add_argument("--num_locality", type=int, default=10, help="base codes N for the locality pairs")
    gen.add_argument("--gaps", type=int, nargs="+", default=[1, 10, 100])
    gen.add_argument("--prompt_file", default=os.path.join(REPO, "test_prompts.txt"))
    gen.add_argument("--seed", type=int, default=1024, help="diffusion seed, shared by all images")
    gen.add_argument("--code_seed", type=int, default=0, help="seed for picking the style codes")
    gen.add_argument("--no_accelerate", action="store_true", help="40-step generation (~20 min per image here)")
    gen.add_argument("--batch_size", type=int, default=4, help="codes generated together per forward pass")

    sc = sub.add_parser("score", help="compute the CSD metrics for generated images")
    sc.add_argument("--out", default="outputs/csd_eval")

    args, passthrough = parser.parse_known_args()
    if args.command == "generate":
        generate(args, passthrough)
    else:
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        score(args)
