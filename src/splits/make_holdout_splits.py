"""Build dose holdouts using only experimental geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.splits import dose_random_in_range
from src.train import train_base

ROOT = Path(__file__).resolve().parents[2]


def input_digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build_splits(frame, dataset, tasks, pair_seed, random_seed, validation_fraction):
    centre = train_base.make_split(frame, "INTERP", seed=pair_seed, val_frac=validation_fraction)
    groups = {
        name: sorted(frame.iloc[indices].PAIR.astype(str).unique().tolist())
        for name, indices in zip(("validation_pairs", "test_pairs"), centre[1:])
    }
    for task, protocol in tasks.items():
        if protocol == "INTERP":
            indices = centre
        elif protocol == "RANDOM_IN_RANGE":
            indices, _ = dose_random_in_range.make_split(frame, dataset, groups, random_seed)
        else:
            indices = train_base.make_split(frame, protocol, seed=pair_seed, val_frac=validation_fraction)
        yield task, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default_benchmark.yaml")
    parser.add_argument("--follow-up", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets = cfg["datasets"][-1:] if args.follow_up else cfg["datasets"]
    tasks = cfg["tasks"]
    if args.follow_up:
        tasks = {name: tasks[name] for name in ("centre", "random")}
    prefix = "followup_" if args.follow_up else ""
    for dataset in datasets:
        points = args.data_dir / dataset / "pairs.parquet"
        digest = input_digest(points)
        frame = pd.read_parquet(points,
                                columns=dose_random_in_range.GEOMETRY_COLUMNS)
        for task, indices in build_splits(frame, dataset, tasks, cfg[prefix + "pair_seed"],
                                          cfg[prefix + "random_seed"], cfg["validation_fraction"]):
            path = args.output_dir / ("followup" if args.follow_up else "original") / dataset / task
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path / "split_indices.npz", **dict(zip(("train", "val", "test"), indices)))
            (path / "split_metadata.json").write_text(json.dumps({
                "input_sha256": digest,
                "indices_sha256": input_digest(path / "split_indices.npz"),
                "task": task,
                "pair_seed": cfg[prefix + "pair_seed"],
                "random_seed": cfg[prefix + "random_seed"],
                "validation_fraction": cfg["validation_fraction"],
            }, indent=2) + "\n", encoding="utf-8")
            print(dataset, task, *(len(index) for index in indices))


if __name__ == "__main__":
    main()
