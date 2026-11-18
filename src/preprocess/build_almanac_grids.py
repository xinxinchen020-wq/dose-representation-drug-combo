"""Reconstruct complete NCI-ALMANAC experiment dose grids while merging only exact duplicates."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess.prep_common import add_dose_position

RULE_VERSION = "almanac-complete-experiment-grid-v1"
CURVE_KEYS = ["NSC1", "NSC2", "CELLNAME"]
BLOCK_KEYS = CURVE_KEYS + ["SCREENER", "STUDY", "TESTDATE", "SOURCE_FIRST_DRUG", "SOURCE_SAMPLE1", "SOURCE_SAMPLE2"]
DOSES = ["CONC1_M", "CONC2_M"]
ORDER_KEYS = ["TESTDATE_ORDER", "SCREENER", "STUDY", "SOURCE_FIRST_DRUG", "SOURCE_SAMPLE1", "SOURCE_SAMPLE2"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path, reference):
    columns = ["COMBODRUGSEQ", "SCREENER", "STUDY", "TESTDATE", "PLATE", "NSC1", "NSC2",
               "SAMPLE1", "SAMPLE2", "CONC1", "CONC2", "CONCUNIT1", "CONCUNIT2",
               "PERCENTGROWTH", "PERCENTGROWTHNOTZ", "EXPECTEDGROWTH", "SCORE", "VALID", "PANEL", "CELLNAME"]
    raw = pd.read_csv(path, usecols=columns, low_memory=False)
    raw["SOURCE_ROW"] = raw.index + 2
    raw.CELLNAME = raw.CELLNAME.astype(str).str.replace(r"[\x00-\x1f\x7f]", "", regex=True).str.strip()
    drugs = set(map(int, json.loads((reference / "drug_dict.json").read_text())["nsc_to_idx"]))
    cells = set(json.loads((reference / "cell_dict.json").read_text())["cell_to_idx"])
    frame = raw[raw.NSC2.notna() & raw.NSC1.isin(drugs) & raw.NSC2.isin(drugs) & raw.CELLNAME.isin(cells)].copy()
    assert frame.VALID.eq("Y").all()
    assert frame.CONCUNIT1.eq("M").all() and frame.CONCUNIT2.eq("M").all()
    assert np.isfinite(frame[["CONC1", "CONC2", "PERCENTGROWTH"]]).all().all()
    assert frame[["CONC1", "CONC2"]].gt(0).all().all()
    assert not frame.COMBODRUGSEQ.duplicated().any()
    frame["SOURCE_FIRST_DRUG"] = frame.NSC1.astype("int64")
    frame["SOURCE_SAMPLE1"] = frame.SAMPLE1.astype("int64")
    frame["SOURCE_SAMPLE2"] = frame.SAMPLE2.astype("int64")
    frame.NSC1 = frame.NSC1.astype("int64")
    frame.NSC2 = frame.NSC2.astype("int64")
    swapped = frame.NSC1 > frame.NSC2
    for first, second in (("NSC1", "NSC2"), ("CONC1", "CONC2"), ("CONCUNIT1", "CONCUNIT2")):
        left = frame.loc[swapped, first].copy()
        frame.loc[swapped, first] = frame.loc[swapped, second]
        frame.loc[swapped, second] = left
    frame = frame.rename(columns={"CONC1": "CONC1_M", "CONC2": "CONC2_M"})
    frame["TESTDATE_ORDER"] = pd.to_datetime(frame.TESTDATE, format="%m/%d/%Y", errors="raise")
    assert not frame[BLOCK_KEYS].isna().any().any()
    return frame, dict(raw_rows=len(raw), retained_raw_rows=len(frame), source_unit="M", raw_sha256=sha(path))


def select_experiment_grids(frame):
    """Select complete experimental dose grids strictly from metadata and grid shape."""
    frame = frame.copy()
    frame["BLOCK_ID"] = pd.factorize(pd.MultiIndex.from_frame(frame[BLOCK_KEYS]), sort=True)[0]
    if frame.duplicated(["BLOCK_ID"] + DOSES).any():
        raise ValueError("Duplicate dose points within the same experimental block and drug ordering")
    blocks = frame.groupby("BLOCK_ID", sort=True).agg(
        n_points=("CONC1_M", "size"),
        doses1=("CONC1_M", lambda values: tuple(sorted(values.unique()))),
        doses2=("CONC2_M", lambda values: tuple(sorted(values.unique()))))
    metadata = frame[BLOCK_KEYS + ["BLOCK_ID", "TESTDATE_ORDER"]].drop_duplicates("BLOCK_ID").set_index("BLOCK_ID")
    blocks = blocks.join(metadata)
    blocks["n1"] = blocks.doses1.map(len)
    blocks["n2"] = blocks.doses2.map(len)
    blocks["complete"] = blocks.n_points.eq(blocks.n1 * blocks.n2) & blocks.n1.ge(3) & blocks.n2.ge(3)
    complete = blocks[blocks.complete].sort_values(CURVE_KEYS + ORDER_KEYS, kind="stable")
    anchors = complete.drop_duplicates(CURVE_KEYS).reset_index()[CURVE_KEYS + ["BLOCK_ID", "SCREENER", "doses1", "doses2"]]
    anchors = anchors.rename(columns={"BLOCK_ID": "anchor_block", "SCREENER": "anchor_screener", "doses1": "anchor_doses1", "doses2": "anchor_doses2"})
    decisions = blocks.reset_index().merge(anchors, on=CURVE_KEYS, how="left", validate="many_to_one")
    if decisions.anchor_block.isna().any():
        raise ValueError("Missing complete original experiment for at least one surface")
    decisions["selected"] = (decisions.complete & decisions.SCREENER.eq(decisions.anchor_screener)
                             & decisions.doses1.eq(decisions.anchor_doses1) & decisions.doses2.eq(decisions.anchor_doses2))
    selected_ids = decisions.loc[decisions.selected, "BLOCK_ID"]
    frame["SELECTED"] = frame.BLOCK_ID.isin(selected_ids)
    selected = frame[frame.SELECTED]
    point_keys = CURVE_KEYS + DOSES
    points = selected.groupby(point_keys, sort=True, as_index=False).agg(
        PERCENTGROWTH=("PERCENTGROWTH", "mean"), EXPECTEDGROWTH=("EXPECTEDGROWTH", "mean"),
        SCORE=("SCORE", "mean"), SCREENER=("SCREENER", "first"), PANEL=("PANEL", "first"),
        N_EXPERIMENTS=("BLOCK_ID", "nunique"))
    return points, decisions, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference, out = args.reference.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    frame, report = load_raw(args.raw, reference)
    points, decisions, provenance = select_experiment_grids(frame)
    assert decisions.complete.all()
    assert set(zip(decisions.n1, decisions.n2)) <= {(3, 3), (3, 5), (5, 3)}
    cells = json.loads((reference / "cell_dict.json").read_text())["cell_to_idx"]
    drugs = json.loads((reference / "drug_dict.json").read_text())["nsc_to_idx"]
    points["PAIR"] = points.NSC1.astype(str) + "_" + points.NSC2.astype(str)
    points["CELL_IDX"] = points.CELLNAME.map(cells).astype("int32")
    for axis in (1, 2):
        points[f"DRUG{axis}_IDX"] = points[f"NSC{axis}"].astype(str).map(drugs).astype("int32")
        points[f"CONC{axis}"] = points[f"CONC{axis}_M"] * 1e6
        points[f"CONC{axis}_LOG"] = np.log10(points[f"CONC{axis}"])
    points = add_dose_position(points)
    for name in ("PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE"):
        points[name] = points[name].astype("float32")
    assert set(points.CELLNAME) == set(cells)
    assert (set(points.NSC1) | set(points.NSC2)) == set(map(int, drugs))
    assert not points.duplicated(CURVE_KEYS + DOSES).any()
    out.mkdir(parents=True)
    features = ["cell_expression.npy", "cell_expression_names.json", "drug_dict.json", "drug_reprs.npz", "drug_smiles.json", "cell_dict.json"]
    for name in features:
        shutil.copy2(reference / name, out / name)
        assert sha(reference / name) == sha(out / name)
    points.to_parquet(out / "pairs.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / "experiment_grid_decisions.parquet", index=False, compression="zstd")
    provenance.to_parquet(out / "raw_experiment_measurements.parquet", index=False, compression="zstd")
    conflicts = decisions.groupby(CURVE_KEYS).selected.agg(["all", "sum", "size"])
    report.update(
        rule_version=RULE_VERSION, dataset="NCI-ALMANAC", raw_source=str(args.raw.resolve()),
        reference=str(reference), reference_sha256=sha(reference / "pairs.parquet"),
        dataset_sha256=sha(out / "pairs.parquet"), feature_hashes={name: sha(out / name) for name in features},
        n_cells=len(cells), n_drugs=len(drugs), n_pairs=int(points.PAIR.nunique()),
        n_curves=int(points.groupby(CURVE_KEYS).ngroups), main_table_rows=len(points),
        original_experiments=len(decisions), selected_experiments=int(decisions.selected.sum()),
        selected_raw_rows=int(provenance.SELECTED.sum()), omitted_raw_rows=int((~provenance.SELECTED).sum()),
        different_grid_curves=int((~conflicts['all']).sum()), curves_merging_matching_experiments=int(conflicts['sum'].gt(1).sum()),
        shape_counts=points.groupby(CURVE_KEYS).agg(n1=('CONC1', 'nunique'), n2=('CONC2', 'nunique')).value_counts().reset_index(name='curves').to_dict('records'),
        target_definition="Original PERCENTGROWTH using time-zero baseline; equal mean over exact matching experiment grids",
        dose_definition="log10(concentration in uM); original M converted via multiplication by 1e6",
        block_definition=BLOCK_KEYS, plate_ids_are_provenance_not_experiment_boundaries=True,
        selection_rule="Earliest test date, then screener/study/original first drug/sample numbers; combine only exact matching dose grids from the same screener",
        other_experiments_retained_in="raw_experiment_measurements.parquet",
    )
    (out / "preprocess_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
