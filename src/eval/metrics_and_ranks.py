"""Summarise validation-selected test predictions under each comparison condition."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

REFERENCE = "raw64_fixed"


def residuals(path):
    with np.load(path, allow_pickle=False) as data:
        y = data["y_true"].astype(np.float64)
        pred = data["y_pred"].astype(np.float64)
        keys = pd.DataFrame({
            "first": data["drug1_key"], "second": data["drug2_key"],
        })
        return data["row_index"], y, y - pred, keys


def matched_errors(candidate, reference):
    rows, y, error, pairs = residuals(candidate)
    ref_rows, ref_y, ref_error, ref_pairs = residuals(reference)
    np.testing.assert_array_equal(rows, ref_rows)
    np.testing.assert_array_equal(y, ref_y)
    pd.testing.assert_frame_equal(pairs, ref_pairs)
    return error, ref_error, pairs


def selected_predictions(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        json.loads((path.parent / item).read_text(encoding="utf-8"))["args"]["seed"]:
        (path.parent / item).with_name("test_pred.npz")
        for item in result["results"]
    }


def pair_rmse(error, pairs):
    return np.sqrt(pairs.assign(squared=error ** 2).groupby(["first", "second"]).squared.mean())


def evaluate(candidate, reference):
    runs = selected_predictions(candidate)
    refs = selected_predictions(reference)
    if runs.keys() != refs.keys():
        raise ValueError(f"Unmatched initialisation seeds for {candidate}")
    overall, paired = [], []
    for seed, path in sorted(runs.items()):
        error, ref_error, pairs = matched_errors(path, refs[seed])
        cr, rr = pair_rmse(error, pairs), pair_rmse(ref_error, pairs)
        paired.append(cr - rr)
        overall.append((np.sqrt(np.mean(error ** 2)), np.mean(np.abs(error)), cr.mean()))
    values = np.array(overall)
    means = values.mean(axis=0)
    return dict(rmse=means[0], mae=means[1], pair_equal_rmse=means[2],
                rmse_sd=values[:, 0].std(ddof=1), mae_sd=values[:, 1].std(ddof=1),
                fraction_pairs_lower=(pd.concat(paired, axis=1).mean(axis=1) < 0).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.csv"))
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    records = []
    manifest = "selected_extended.json" if args.extended else "selected.json"
    for path in sorted(args.runs.glob(f"*/*/*/*/*/{manifest}")):
        split, dataset, task, architecture, method, _ = path.relative_to(args.runs).parts
        reference = path.parent.parent / REFERENCE / manifest
        records.append(dict(split=split, dataset=dataset, task=task,
                            architecture=architecture, method=method, **evaluate(path, reference)))
    if not records:
        raise ValueError(f"No selected runs found under {args.runs}")
    summary = pd.DataFrame(records)
    for metric in ("rmse", "mae"):
        summary[metric + "_rank"] = summary.groupby(
            ["split", "dataset", "task", "architecture"])[metric].rank(method="min")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
