"""Write run_pope configs for the online entropy-routing methods (one per condition).

    python -m scripts.make_dola_configs topk --top-k 3 --temperature 0.5
    python -m scripts.make_dola_configs gradient --patience 1 --epsilon 0.01
    python -m scripts.make_dola_configs soft

Writes configs/dola/<family>/<name>.yaml for clean + blur s1-4 + noise s1-4. Only needed to decode
full answers online; AUROC/F1 of every routing variant can be computed offline from the layer
readouts (scripts/run_layers.py + scripts/analyze_routing.py), which is how hyperparameters
should be chosen (on DEV images).
"""
import argparse

import yaml

from scripts.run_layers import CONDITIONS
from src.paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["topk", "soft", "gradient"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--decoding", default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.family == "topk":
        method, margs, tag = "dola_topk", {"top_k": args.top_k, "temperature": args.temperature}, f"k{args.top_k}_t{args.temperature:g}"
    elif args.family == "soft":
        method, margs, tag = "dola_topk", {"top_k": 7, "temperature": args.temperature}, f"t{args.temperature:g}"
    else:
        method, margs, tag = "dola_gradient", {"patience": args.patience, "epsilon": args.epsilon}, f"p{args.patience}_e{args.epsilon:g}"

    out_dir = REPO_ROOT / "configs" / "dola" / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, level in CONDITIONS:
        name = f"dola_{args.family}_{tag}_{args.decoding}_{kind}_s{level}"
        cfg = {"name": name, "method": method, "method_args": margs, "decoding": args.decoding,
               "seed": args.seed, "kind": kind, "level": level, "max_new_tokens": 16}
        if args.limit:
            cfg["limit"] = args.limit
        with open(out_dir / f"{name}.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {len(CONDITIONS)} configs to {out_dir}")


if __name__ == "__main__":
    main()
