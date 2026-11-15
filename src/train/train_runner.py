"""Run the manuscript benchmark and select learning rates on validation error."""
import argparse
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

import yaml

from src.splits.make_holdout_splits import input_digest

ROOT = Path(__file__).resolve().parents[2]


def groups(cfg, stage):
    datasets = cfg["datasets"]
    if stage == "primary":
        yield from product(["original"], datasets, cfg["tasks"],
                           ["joint_concatenation"], cfg["methods"])
    else:
        tasks = ("centre", "random")
        joint, fusion = "joint_concatenation", "within_drug_fusion"
        yield from product(["original"], datasets, tasks, [joint, fusion], cfg["focal_methods"])
        yield from product(["original"], datasets[-1:], tasks, [joint], [cfg["compact_method"]])
        yield from product(["followup"], datasets[-1:], tasks, [joint], cfg["focal_methods"])


def read_result(directory):
    results = list(directory.glob("*/results.json"))
    if len(results) != 1:
        raise RuntimeError(f"Expected one completed fit in {directory}")
    return results[0], json.loads(results[0].read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["primary", "extended"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default_benchmark.yaml")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage_groups = list(groups(cfg, args.stage))
    if args.plan:
        print(f"{len(stage_groups)} comparison groups")
        for group in stage_groups:
            print(*group)
        return

    dataset_digests = {}
    for split, dataset, task, architecture, method in stage_groups:
        group = args.output_dir / split / dataset / task / architecture / method
        data = args.data_dir / dataset
        indices = args.split_dir / split / dataset / task / "split_indices.npz"
        if not indices.is_file():
            raise FileNotFoundError(indices)
        if dataset not in dataset_digests:
            dataset_digests[dataset] = input_digest(data / "pairs.parquet")
        metadata = json.loads(indices.with_name("split_metadata.json").read_text(encoding="utf-8"))
        prefix = "followup_" if split == "followup" else ""
        if (metadata["input_sha256"] != dataset_digests[dataset]
                or metadata["indices_sha256"] != input_digest(indices)
                or metadata["task"] != task
                or metadata["pair_seed"] != cfg[prefix + "pair_seed"]
                or metadata["random_seed"] != cfg[prefix + "random_seed"]
                or metadata["validation_fraction"] != cfg["validation_fraction"]):
            raise ValueError(f"Regenerate the holdout for {dataset} {split} {task}")

        def fit(rate, seed):
            output = group / f"lr-{rate:g}" / f"seed-{seed}"
            signature = {
                "input_sha256": metadata["input_sha256"],
                "indices_sha256": metadata["indices_sha256"],
                "architecture": cfg["architectures"][architecture],
                "protocol": cfg["tasks"][task], "method": method, "seed": seed, "rate": rate,
                "epochs": cfg["epochs"], "patience": cfg["patience"],
                "batch_size": cfg["batch_size"], "rff_sigma": cfg["rff_sigma"],
            }
            record = output / "fit_configuration.json"
            if list(output.glob("*/results.json")) and (
                    not record.is_file() or json.loads(record.read_text(encoding="utf-8")) != signature):
                raise ValueError(f"Existing fit has different inputs or settings: {output}")
            if not list(output.glob("*/results.json")):
                command = [
                    sys.executable, "-m", "src.train.run_fit", "--indices", str(indices),
                    "--data", str(data), "--out", str(output),
                    "--dose-enc", method, "--arch", cfg["architectures"][architecture],
                    "--protocol", cfg["tasks"][task], "--seed", str(seed),
                    "--split-seed", str(cfg["pair_seed"] if split == "original" else cfg["followup_pair_seed"]),
                    "--val-frac", str(cfg["validation_fraction"]), "--shared-backbone-init",
                    "--batch-size", str(cfg["batch_size"]), "--epochs", str(cfg["epochs"]),
                    "--patience", str(cfg["patience"]), "--sigma", str(cfg["rff_sigma"]),
                    "--lr", str(rate), "--device", args.device, "--save-pred",
                ]
                subprocess.run(command, check=True)
                record.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
            path, result = read_result(output)
            error = result["best_val_rmse"]
            if not math.isfinite(error):
                raise ValueError(f"Non-finite validation error in {path}")
            return path, error

        seeds = cfg["initialisation_seeds"]
        candidates = {rate: [fit(rate, seed) for seed in seeds] for rate in cfg["learning_rates"]}
        scores = {rate: mean(error for _, error in fits) for rate, fits in candidates.items()}
        best = next(rate for rate in cfg["tie_order"] if scores[rate] <= min(scores.values()) + cfg["tie_tolerance"])
        selected = candidates[best]
        if args.stage == "extended" and split == "original":
            selected = selected + [fit(best, seed) for seed in cfg["additional_seeds"]]
        summary = {
            "learning_rate": best, "validation_means": scores,
            "results": [path.relative_to(group).as_posix() for path, _ in selected],
        }
        manifest = "selected.json" if args.stage == "primary" else "selected_extended.json"
        (group / manifest).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
