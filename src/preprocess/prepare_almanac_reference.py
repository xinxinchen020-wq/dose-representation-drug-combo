"""Prepare NCI-ALMANAC reference dataset with aligned DepMap or CellMiner expression profiles."""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess.prep_common import (
    add_dose_position, jsonable, load_alias_map, load_cellminer,
    load_depmap, load_gene_panel, load_hgnc, norm,
)

ROOT = str(Path(__file__).resolve().parents[2])

DROP_NSC = {753082}
CLEAN_RE = re.compile(r"[\x00-\x1f\x7f]")

CM_XLS = "RNA__RNA_seq_composite_expression.xls"
DEPMAP_CSV = "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
HGNC_TSV = "hgnc_complete_set.txt"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_combo(raw_dir):
    for c in ["NCL_Combo_Response.csv", "NCL_Combo_Response.csv.gz",
              "ComboDrugGrowth_Nov2017.csv", "ComboDrugGrowth_Nov2017.csv.gz"]:
        f = os.path.join(raw_dir, c)
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"NCI-ALMANAC combination response file not found in {raw_dir}")


def find_smiles(raw_dir):
    for c in ["nsc_smiles.csv", "nsc_smiles.csv.gz"]:
        f = os.path.join(raw_dir, c)
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"nsc_smiles.csv not found in {raw_dir}")


def load_and_clean(combo_csv, smi_csv, report):
    log(f"Reading {os.path.basename(combo_csv)} ...")
    usecols = ["NSC1", "NSC2", "CONC1", "CONC2", "CONCUNIT1", "CONCUNIT2",
               "CONCINDEX1", "CONCINDEX2",
               "PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE", "CELLNAME", "PANEL", "SCREENER"]
    df = pd.read_csv(combo_csv, usecols=usecols, low_memory=False)
    log(f"  {len(df):,} rows")
    report["rows_raw"] = len(df)

    n_ctrl = int(df["CELLNAME"].astype(str).str.contains(CLEAN_RE, regex=True).sum())
    df["CELLNAME"] = df["CELLNAME"].astype(str).map(lambda s: CLEAN_RE.sub("", s).strip())
    report["cellname_control_chars"] = n_ctrl

    df = df[df["NSC2"].notna()].copy()

    smi = pd.read_csv(smi_csv)
    smi["NSC"] = pd.to_numeric(smi["NSC"], errors="coerce")
    smi_nsc = set(smi["NSC"].dropna().astype(int))
    used = set(pd.to_numeric(df["NSC1"], errors="coerce").dropna().astype(int)) | \
           set(pd.to_numeric(df["NSC2"], errors="coerce").dropna().astype(int))
    report["drugs_used"] = len(used)
    report["drugs_no_smiles"] = sorted(used - smi_nsc)
    valid = (used & smi_nsc) - DROP_NSC

    n1 = pd.to_numeric(df["NSC1"], errors="coerce").astype("Int64")
    n2 = pd.to_numeric(df["NSC2"], errors="coerce").astype("Int64")
    keep = n1.isin(valid) & n2.isin(valid)
    report["rows_dropped_no_smiles"] = int((~keep).sum())
    df = df[keep].copy()
    df["NSC1"], df["NSC2"] = n1[keep].astype(int), n2[keep].astype(int)

    # Canonical drug order: NSC1 <= NSC2
    sw = df["NSC1"] > df["NSC2"]
    for a, b in [("NSC1", "NSC2"), ("CONC1", "CONC2"), ("CONCINDEX1", "CONCINDEX2")]:
        va, vb = df.loc[sw, a].copy(), df.loc[sw, b].copy()
        df.loc[sw, a], df.loc[sw, b] = vb, va
    report["rows_swapped"] = int(sw.sum())
    return df, smi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-source", choices=["cellminer", "depmap"], required=True)
    ap.add_argument("--cell-subset", default=None,
                    help="JSON listing NCI-ALMANAC cell lines to retain")
    ap.add_argument("--n-genes", type=int, default=1000)
    ap.add_argument("--gene-panel", default=None,
                    help="Gene panel JSON path or 'variance' for top-variance fallback (default: configs/gene_panel_l1000.json)")
    ap.add_argument("--config", default=None,
                    help="Configuration label for report (defaults to basename of --out)")
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--configs-dir", default=os.path.join(ROOT, "configs"))
    ap.add_argument("--processed-dir", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--skip-expr", action="store_true")
    args = ap.parse_args()

    if args.gene_panel is None:
        args.gene_panel = os.path.join(ROOT, "configs", "gene_panel_l1000.json")

    cfg = args.config or os.path.basename(os.path.normpath(args.out))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log(f"Config={cfg} cell_source={args.cell_source}")

    maps = json.load(open(os.path.join(args.configs_dir, "cell_maps.json"), encoding="utf-8"))
    alm_info = maps["almanac"]

    subset = None
    if args.cell_subset:
        subset = set(json.load(open(args.cell_subset, encoding="utf-8"))["cells"])
        log(f"Cell subset: {len(subset)} cell lines")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg, "dataset": "NCI-ALMANAC",
        "cell_source": ("CellMiner RNA-seq (log2(FPKM+1))"
                        if args.cell_source == "cellminer"
                        else "DepMap 25Q3 RNA-seq (log2(TPM+1))"),
        "target_column": "PERCENTGROWTH",
        "split_ratio": "60/20/20",
        "args": {k: v for k, v in vars(args).items()},
    }

    df, smi = load_and_clean(find_combo(os.path.join(args.raw_dir, "almanac")),
                             find_smiles(os.path.join(args.raw_dir, "almanac")), report)

    # Convert NCI-ALMANAC molar concentrations to micromolar (x1e6) so both screens
    # share identical log10(uM) units.
    units = df["CONCUNIT1"].dropna().astype(str).unique()
    report["conc_unit_raw"] = df["CONCUNIT1"].astype(str).value_counts().to_dict()
    if set(units) - {"M"}:
        raise ValueError(f"Unexpected concentration unit in NCI-ALMANAC: {units}")
    M_TO_UM = 1e6
    for c in ["CONC1", "CONC2"]:
        v = pd.to_numeric(df[c], errors="coerce") * M_TO_UM
        df[c] = v
        df[c + "_LOG"] = np.log10(v.where(v > 0))
    report["conc_unit"] = "uM (converted from M via *1e6)"
    report["conc_log_range"] = [float(np.nanmin(df["CONC1_LOG"])),
                                float(np.nanmax(df["CONC1_LOG"]))]
    log(f"  Concentrations converted M -> uM; log10(uM) in "
        f"[{report['conc_log_range'][0]:.2f}, {report['conc_log_range'][1]:.2f}]")
    for c in ["PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["PAIR"] = df["NSC1"].astype(str) + "_" + df["NSC2"].astype(str)
    df = add_dose_position(df)
    log(f"  Retained {len(df):,} rows")

    if not args.skip_expr:
        hgnc = os.path.join(args.raw_dir, "gene_lists", HGNC_TSV)
        panel = amap = None
        if args.gene_panel != "variance":
            panel, _pj = load_gene_panel(args.gene_panel)
            amap = load_alias_map(hgnc)
            log(f"Gene panel {_pj['panel']}: {len(panel)} genes")
            report["gene_panel"] = {
                "name": _pj["panel"], "n_genes": len(panel),
                "source": _pj.get("source", ""), "file": args.gene_panel,
            }
        else:
            pc_entrez, pc_syms = load_hgnc(hgnc)
            report["protein_coding_entrez"] = len(pc_entrez)
            report["gene_panel"] = {"name": "variance_top", "n_genes": args.n_genes}

        if args.cell_source == "cellminer":
            p = os.path.join(args.raw_dir, "expression", CM_XLS)
            log(f"Reading CellMiner: {CM_XLS}")
            labels, X, genes = load_cellminer(
                p, args.n_genes,
                pc_entrez if args.gene_panel == "variance" else None,
                pc_syms if args.gene_panel == "variance" else None,
                panel=panel, amap=amap)
            key_of = lambda name: alm_info.get(name, {}).get("cellminer", "")
        else:
            p = os.path.join(args.raw_dir, "depmap", DEPMAP_CSV)
            log(f"Reading DepMap: {DEPMAP_CSV}")
            labels, X, genes = load_depmap(
                p, args.n_genes,
                pc_entrez if args.gene_panel == "variance" else None,
                pc_syms if args.gene_panel == "variance" else None,
                panel=panel, amap=amap)
            key_of = lambda name: alm_info.get(name, {}).get("ach", "")
        log(f"  Expression shape={X.shape} ({len(labels)} cells x {len(genes)} genes)")
        lab_idx = {l: i for i, l in enumerate(labels)}
        report["expression_shape"] = [int(x) for x in X.shape]
        report["genes_top5"] = list(genes[:5])
    else:
        labels, X, genes, lab_idx = [], None, [], {}
        key_of = lambda name: ""

    alm_cells = sorted(df["CELLNAME"].unique(), key=norm)
    if subset is not None:
        alm_cells = [c for c in alm_cells if c in subset]
    mapping, unmapped = {}, []
    for name in alm_cells:
        k = key_of(name)
        if k and k in lab_idx:
            mapping[name] = k
        else:
            unmapped.append(name)
    report["cell_lines_considered"] = len(alm_cells)
    report["cell_lines_mapped"] = len(mapping)
    report["cell_lines_unmapped"] = unmapped
    log(f"  Cell mapping: {len(mapping)}/{len(alm_cells)}"
        + (f" (unmapped={unmapped})" if unmapped else ""))

    df = df[df["CELLNAME"].isin(mapping)].copy()
    report["rows_after_cell_filter"] = len(df)

    cells_used = sorted(df["CELLNAME"].unique())
    cell2idx = {c: i for i, c in enumerate(cells_used)}
    df["CELL_IDX"] = df["CELLNAME"].map(cell2idx)
    drugs = sorted(set(df["NSC1"]) | set(df["NSC2"]))
    d2i = {n: i for i, n in enumerate(drugs)}
    df["DRUG1_IDX"] = df["NSC1"].map(d2i)
    df["DRUG2_IDX"] = df["NSC2"].map(d2i)
    report["n_cells"] = len(cells_used)
    report["n_drugs"] = len(drugs)
    report["n_pairs"] = int(df.groupby(["NSC1", "NSC2"]).ngroups)

    if X is not None:
        order = [lab_idx[mapping[c]] for c in cells_used]
        np.save(os.path.join(args.out, "cell_expression.npy"), X[order])
        json.dump({"cell_names": cells_used, "genes": genes},
                  open(os.path.join(args.out, "cell_expression_names.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
        log(f"  Saved expression array shape={X[order].shape}")

    out = df[["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX", "CONC1", "CONC2",
              "CONC1_LOG", "CONC2_LOG", "CONC1_LOG_POS", "CONC2_LOG_POS",
              "PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE",
              "SCREENER", "PANEL", "PAIR", "NSC1", "NSC2", "CELLNAME"]].copy()
    for c in ["PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE", "CONC1", "CONC2",
              "CONC1_LOG", "CONC2_LOG", "CONC1_LOG_POS", "CONC2_LOG_POS"]:
        out[c] = out[c].astype(np.float32)
    for c in ["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX"]:
        out[c] = out[c].astype(np.int32)
    out.to_parquet(os.path.join(args.out, "pairs.parquet"), index=False, compression="zstd")
    report["main_table_rows"] = len(out)

    json.dump({"cell_to_idx": cell2idx, "idx_to_cell": {str(i): c for c, i in cell2idx.items()},
               "cell_key": mapping},
              open(os.path.join(args.out, "cell_dict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"nsc_to_idx": {str(n): i for n, i in d2i.items()},
               "idx_to_nsc": {str(i): int(n) for n, i in d2i.items()}},
              open(os.path.join(args.out, "drug_dict.json"), "w", encoding="utf-8"), indent=1)
    smi["NSC"] = pd.to_numeric(smi["NSC"], errors="coerce")
    s_map = dict(zip(smi["NSC"].astype("Int64"), smi["SMILES"]))
    json.dump({"order": [int(n) for n in drugs], "smiles": [s_map.get(n) for n in drugs]},
              open(os.path.join(args.out, "drug_smiles.json"), "w", encoding="utf-8"), indent=1)

    report["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(jsonable(report), open(os.path.join(args.out, "preprocess_report.json"), "w",
                                     encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

    log(f"Finished [{cfg}] rows={report['main_table_rows']:,} cells={report['n_cells']} "
        f"drugs={report['n_drugs']} pairs={report['n_pairs']} elapsed={report['elapsed_sec']}s")


if __name__ == "__main__":
    main()
