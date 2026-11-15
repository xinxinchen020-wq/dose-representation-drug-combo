"""Analyse matched prediction errors by drug exclusion and by drug group."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics_and_ranks import matched_errors, selected_predictions


def analyse(candidate, reference, focal_drug, exclude):
    runs, refs = selected_predictions(candidate), selected_predictions(reference)
    if runs.keys() != refs.keys():
        raise ValueError(f"Unmatched initialisation seeds for {candidate}")
    records = []
    for seed, path in sorted(runs.items()):
        error, ref_error, pairs = matched_errors(path, refs[seed])
        full = np.sqrt(np.mean(error ** 2)) - np.sqrt(np.mean(ref_error ** 2))
        drugs = sorted(set(pairs["first"]) | set(pairs["second"])) if exclude else [focal_drug]
        for drug in drugs:
            contains = (pairs["first"] == drug) | (pairs["second"] == drug)
            keep = ~contains.to_numpy()
            if not keep.any():
                continue
            sq_difference = error ** 2 - ref_error ** 2
            records.append(dict(
                seed=seed, drug=drug, pairs_removed=len(pairs[contains].drop_duplicates()),
                points_removed=int(contains.sum()), full_rmse_difference=full,
                excluded_rmse_difference=np.sqrt(np.mean(error[keep] ** 2)) -
                                         np.sqrt(np.mean(ref_error[keep] ** 2)),
                focal_rmse=np.sqrt(np.mean(error[~keep] ** 2)) if contains.any() else np.nan,
                other_rmse=np.sqrt(np.mean(error[keep] ** 2)),
                focal_mse_contribution=sq_difference[~keep].sum() / len(error),
                other_mse_contribution=sq_difference[keep].sum() / len(error),
            ))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/drug-analysis"))
    args = parser.parse_args()
    root = args.runs / "original/almanac_dm47_batch_checked"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in ("centre", "random"):
        condition = root / task / "joint_concatenation"
        reference = condition / "raw64_fixed/selected.json"
        records, diagnostics = [], []
        for path in sorted(condition.glob("*/selected.json")):
            method = path.parent.name
            records.extend(dict(method=method, **row) for row in analyse(path, reference, "123127", False))
            if task == "centre" and method in {"linear64_mlp", "D3"}:
                diagnostics.extend(dict(method=method, **row) for row in analyse(path, reference, "123127", True))
        if not records:
            raise ValueError(f"No validation-selected fits at {condition}")
        pd.DataFrame(records).to_csv(args.output_dir / f"{task}_group_contributions.csv", index=False)
        if diagnostics:
            pd.DataFrame(diagnostics).to_csv(args.output_dir / "centre_drug_exclusion.csv", index=False)


if __name__ == "__main__":
    main()
