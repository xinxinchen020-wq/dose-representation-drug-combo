"""按原始批次的完整剂量网格构建独立数据版本，复用已有药物与细胞特征。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CURVE_KEYS = ["CELLNAME", "NSC1", "NSC2"]
RAW_NAMES = ["BATCH_ID", "CELLNAME", "NSC1", "CONC1", "NSC2", "CONC2",
             "COMBINATION_NAME", "viability1", "viability2", "viability3", "viability4",
             "MU_NORM", "X_X0"]
RULE_VERSION = "oneil-complete-batch-grid-v1"


def load_raw(path):
    frame = pd.read_csv(path)
    if frame.columns[0] != "BatchID" or len(frame.columns) != len(RAW_NAMES):
        raise ValueError("原始批次列或列数不符合O’Neil补充材料")
    frame.columns = RAW_NAMES
    frame["SOURCE_ROW"] = np.arange(2, len(frame) + 2)  # 原始工作簿行号，含表头。
    swapped = frame.NSC1 > frame.NSC2
    for first, second in (("NSC1", "NSC2"), ("CONC1", "CONC2")):
        values = frame.loc[swapped, first].copy()
        frame.loc[swapped, first] = frame.loc[swapped, second]
        frame.loc[swapped, second] = values
    values = [f"viability{i}" for i in range(1, 5)]
    for column in values + ["CONC1", "CONC2"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["N_TECHNICAL_REPLICATES"] = frame[values].notna().sum(axis=1)
    frame["PERCENTGROWTH"] = 100 * frame[values].mean(axis=1)
    if not np.isfinite(frame[["CONC1", "CONC2", "PERCENTGROWTH"]]).all().all():
        raise ValueError("原始数据存在缺失或非有限剂量、响应，需要单独处理")
    if (frame[["CONC1", "CONC2"]] <= 0).any().any():
        raise ValueError("本版本仅支持严格正浓度")
    return frame


def select_complete_batches(frame):
    """只按批次编号和完整网格决定保留批次，不按响应或测试误差决定。"""
    block_keys = CURVE_KEYS + ["BATCH_ID"]
    if frame.duplicated(block_keys + ["CONC1", "CONC2"]).any():
        raise ValueError("同一批次存在重复剂量点，不能当作新的技术重复")
    blocks = frame.groupby(block_keys, sort=True).agg(
        rows=("CONC1", "size"),
        doses1=("CONC1", lambda values: tuple(sorted(values.unique()))),
        doses2=("CONC2", lambda values: tuple(sorted(values.unique()))),
    ).reset_index()
    blocks["complete_four_by_four"] = (
        blocks.doses1.map(len).eq(4) & blocks.doses2.map(len).eq(4) & blocks.rows.eq(16))
    decisions = []
    for _, curve in blocks.groupby(CURVE_KEYS, sort=True):
        complete = curve[curve.complete_four_by_four].sort_values("BATCH_ID")
        if complete.empty:
            raise ValueError("有组合和细胞不具备完整四乘四批次，不能静默丢弃")
        anchor = complete.iloc[0]
        for _, block in curve.iterrows():
            keep = (block.complete_four_by_four and block.doses1 == anchor.doses1
                    and block.doses2 == anchor.doses2)
            decisions.append({**{key: block[key] for key in block_keys},
                              "anchor_batch": int(anchor.BATCH_ID), "selected": bool(keep),
                              "doses1": list(block.doses1), "doses2": list(block.doses2),
                              "reason": "exact_grid_match" if keep else "different_or_incomplete_grid"})
    decisions = pd.DataFrame(decisions)
    selected = frame.merge(decisions[block_keys + ["selected"]], on=block_keys, validate="many_to_one")
    selected = selected[selected.selected].drop(columns="selected")
    point_keys = CURVE_KEYS + ["CONC1", "CONC2"]
    points = selected.groupby(point_keys, sort=True, as_index=False).agg(
        PERCENTGROWTH=("PERCENTGROWTH", "mean"),  # 每个批次均值权重相同。
        N_TECHNICAL_REPLICATES=("N_TECHNICAL_REPLICATES", "sum"),
        N_BATCHES=("BATCH_ID", "nunique"),
        SOURCE_BATCHES=("BATCH_ID", lambda values: ",".join(map(str, sorted(values.unique())))),
    )
    return points, decisions, selected


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("请使用新的数据目录，原始数据和旧版本不覆盖")
    reference = pd.read_parquet(args.reference / "pairs.parquet")
    frame = load_raw(args.raw)
    all_raw_rows = len(frame)
    frame = frame[frame.CELLNAME.isin(reference.CELLNAME.unique())].copy()
    points, decisions, selected = select_complete_batches(frame)
    metadata_columns = ["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX", "SCREENER", "PANEL", "PAIR"]
    metadata = reference[CURVE_KEYS + metadata_columns].drop_duplicates()
    if metadata.duplicated(CURVE_KEYS).any():
        raise ValueError("已有实体与特征映射不一致")
    points = points.merge(metadata, on=CURVE_KEYS, how="left", validate="many_to_one")
    if points[metadata_columns].isna().any().any():
        raise ValueError("新剂量表有实体无法复用旧特征")
    for axis in (1, 2):
        column = f"CONC{axis}_LOG"
        points[column] = np.log10(points[f"CONC{axis}"])
        groups = points.groupby(CURVE_KEYS)[column]
        low, high = groups.transform("min"), groups.transform("max")
        points[f"CONC{axis}_LOG_POS"] = (points[column] - low) / (high - low)
    points["EXPECTEDGROWTH"] = np.nan
    points["SCORE"] = np.nan
    for column in ["CONC1", "CONC2", "CONC1_LOG", "CONC2_LOG", "CONC1_LOG_POS",
                   "CONC2_LOG_POS", "PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE"]:
        points[column] = points[column].astype(np.float32)
    for column in ["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX", "N_TECHNICAL_REPLICATES", "N_BATCHES"]:
        points[column] = points[column].astype(np.int32)
    points = points[list(reference.columns) + ["N_TECHNICAL_REPLICATES", "N_BATCHES", "SOURCE_BATCHES"]]
    assert len(points) == metadata.shape[0] * 16
    assert not points.duplicated(CURVE_KEYS + ["CONC1", "CONC2"]).any()
    args.out.mkdir(parents=True)
    copied_hashes = {}
    for source in args.reference.iterdir():
        if source.is_file() and source.name not in ("pairs.parquet", "preprocess_report.json"):
            target = args.out / source.name
            shutil.copy2(source, target)
            assert sha256(source) == sha256(target)
            copied_hashes[source.name] = sha256(target)
    points.to_parquet(args.out / "pairs.parquet", index=False, compression="zstd")
    decisions.to_json(args.out / "batch_decisions.json", orient="records", indent=2)
    frame.to_parquet(args.out / "raw_batch_measurements.parquet", index=False, compression="zstd")
    changed = decisions.groupby(CURVE_KEYS).selected.agg(["size", "sum"])
    report = dict(
        rule_version=RULE_VERSION, raw_source=str(args.raw), raw_sha256=sha256(args.raw),
        reference=str(args.reference), reference_sha256=sha256(args.reference / "pairs.parquet"),
        dataset_sha256=sha256(args.out / "pairs.parquet"), feature_hashes=copied_hashes,
        target_definition="100 × mean of available viability1..4 within batch; equal mean of exactly matching batches",
        dose_definition="log10(concentration in micromolar); original concentration values retained",
        selection_rule="Use lowest-numbered complete batch grid; merge only batches whose two concentration grids exactly match it",
        raw_rows=all_raw_rows, cell_filtered_raw_rows=len(frame), selected_raw_rows=len(selected),
        omitted_raw_rows=len(frame) - len(selected), n_curves=len(metadata), main_table_rows=len(points),
        n_cells=int(points.CELLNAME.nunique()), n_drugs=len(set(points.NSC1) | set(points.NSC2)),
        n_pairs=int(points.PAIR.nunique()), curves_with_grid_conflict=int((changed['size'] != changed['sum']).sum()),
        curves_merging_matching_batches=int((changed['sum'] > 1).sum()),
        selected_replicate_counts={str(k): int(v) for k, v in selected.N_TECHNICAL_REPLICATES.value_counts().sort_index().items()},
        different_grids_are_not_rounded_or_pooled=True,
        other_batches_retained_in="raw_batch_measurements.parquet",
    )
    (args.out / "preprocess_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
