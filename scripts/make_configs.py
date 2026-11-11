"""Generate experiment configs.

    python -m scripts.make_configs stage0
"""
import argparse
from pathlib import Path

import yaml

from src.degradations import KINDS, LEVELS
from src.paths import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs"
SAMPLE_SEEDS = [55, 56, 57]


def stage0() -> list[dict]:
    """Clean images, paper settings (alpha=1, beta=0.1, T=500): sampling over 3 seeds to compare
    with the VCD paper's Table 1, plus one greedy run per method."""
    configs = []
    for method in ["baseline", "vcd"]:
        for seed in SAMPLE_SEEDS:
            configs.append(dict(name=f"stage0_{method}_sample_seed{seed}", method=method, decoding="sample", seed=seed, noise_step=500))
        configs.append(dict(name=f"stage0_{method}_greedy", method=method, decoding="greedy", seed=55, noise_step=500))
    # Repo issue #24 reports the paper's POPE runs used noise_step=999, not the script default 500
    for seed in SAMPLE_SEEDS:
        configs.append(dict(name=f"stage0_vcd_T999_sample_seed{seed}", method="vcd", decoding="sample", seed=seed, noise_step=999))
    configs.append(dict(name="stage0_vcd_T999_greedy", method="vcd", decoding="greedy", seed=55, noise_step=999))
    configs.append(dict(name="stage0_smoke", method="vcd", decoding="sample", seed=55, splits=["random"], limit=30))
    return configs


def stage1() -> list[dict]:
    """Severity-ladder pilot: greedy baseline on a fixed 100-question subset per split, at every
    degradation level. Accuracy should fall monotonically from clean to s4."""
    conditions = [("clean", 0)] + [(kind, level) for kind in KINDS for level in range(1, len(LEVELS[kind]) + 1)]
    return [
        dict(name=f"stage1_baseline_greedy_{kind}_s{level}", method="baseline", decoding="greedy", seed=55,
             kind=kind, level=level, limit=100)
        for kind, level in conditions
    ]


CALIBRATION_PARAMS = {
    "blur": [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20],
    "noise": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
}


def calib() -> list[dict]:
    """Severity calibration sweep: greedy baseline on the Stage 1 subset, degrading on the fly
    with explicit parameters, to choose levels with roughly even accuracy drops."""
    return [
        dict(name=f"calib_{kind}_{param:g}", method="baseline", decoding="greedy", seed=55,
             kind=kind, level=0, param=param, limit=100)
        for kind, params in CALIBRATION_PARAMS.items() for param in params
    ]


def stage2() -> list[dict]:
    """Main grid: baseline vs VCD (T=999) on every condition, full POPE splits, greedy plus
    sampling over 3 seeds. Grouped into one job per (method, decoding/seed, kind)."""
    conditions = [("clean", 0)] + [(kind, level) for kind in KINDS for level in range(1, len(LEVELS[kind]) + 1)]
    runs = [("greedy", 55)] + [("sample", seed) for seed in SAMPLE_SEEDS]
    configs = []
    for method in ["baseline", "vcd"]:
        for decoding, seed in runs:
            tag = f"{method}_{decoding}" + (f"_seed{seed}" if decoding == "sample" else "")
            for kind, level in conditions:
                configs.append(dict(name=f"stage2_{tag}_{kind}_s{level}", method=method, decoding=decoding,
                                    seed=seed, noise_step=999, kind=kind, level=level, _group=f"{tag}_{kind}"))
    return configs


def fixb() -> list[dict]:
    """Phase 2 fix B: the DAE-restored image as the real-branch input. Greedy, noise levels only
    (dae_v1 leaves blurred images unchanged). Compare against the stage2 greedy runs on the same cells.
      b0: baseline on the restored image (control: what restoration alone does)
      b1: VCD on the restored image (contrast = restored + diffusion noise, T=999)
      b2: VCD contrasting the restored image against the original degraded image
    One job per variant."""
    variants = {
        "b0_baseline_dae": dict(method="baseline"),
        "b1_vcd_dae": dict(method="vcd", contrast="diffusion"),
        "b2_vcd_dae_vs_degraded": dict(method="vcd", contrast="degraded"),
    }
    configs = [
        dict(name=f"fixb_{tag}_greedy_noise_s{level}", decoding="greedy", seed=55, noise_step=999,
             kind="noise", level=level, restore="dae_v1", **variant, _group=f"{tag}_greedy")
        for tag, variant in variants.items() for level in range(1, len(LEVELS["noise"]) + 1)
    ]
    configs.append(dict(name="fixb_smoke", method="vcd", decoding="greedy", seed=55, noise_step=999, kind="noise",
                        level=4, restore="dae_v1", contrast="degraded", splits=["random"], limit=30))
    return configs


STAGES = {"stage0": stage0, "stage1": stage1, "calib": calib, "stage2": stage2, "fixb": fixb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    for cfg in STAGES[args.stage]():
        # optional _group puts configs in a subdir, which submit.sh runs as one job
        out_dir = CONFIG_DIR / args.stage / cfg.pop("_group", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{cfg['name']}.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(out_dir / f"{cfg['name']}.yaml")


if __name__ == "__main__":
    main()
