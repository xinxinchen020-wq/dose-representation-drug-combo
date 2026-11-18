"""Prepare O'Neil 2016 reference dataset with DepMap 25Q3 RNA-seq expression profiles."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess.prep_common import (
    add_dose_position, jsonable, load_alias_map, load_depmap,
    load_drug_smiles, load_gene_panel, load_hgnc, norm,
)

ROOT = str(Path(__file__).resolve().parents[2])

DEPMAP_CSV = "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
HGNC_TSV = "hgnc_complete_set.txt"

RAW_COLS = ["cell", "dA", "cA", "dB", "cB", "combo",
            "v1", "v2", "v3", "v4", "mu", "X0"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_panel_map(xlsx_path):
    """Build {depmap_id: tissue_suffix} lookup from DrugComb cell_line metadata."""
    dc = pd.read_excel(xlsx_path, sheet_name="cell_line")
    out = {}
    for _, r in dc.iterrows():
        dm = r.get("depmap_id")
        cc = r.get("ccle_name")
        if pd.isna(dm) or not isinstance(cc, str) or "_" not in cc:
            continue
        out.setdefault(str(dm), cc.split("_", 1)[1])
    return out


def load_oneil(csv_path, report):
    """Load and normalize the raw O'Neil combination table."""
    log(f"Reading {os.path.basename(csv_path)} ...")
    df = pd.read_csv(csv_path, index_col=0)
    if len(df.columns) != len(RAW_COLS):
        raise ValueError(f"Expected {len(RAW_COLS)} columns, found {len(df.columns)}")
    df.columns = RAW_COLS
    log(f"  {len(df):,} rows")
    report["rows_raw"] = len(df)

    # Enforce canonical lexicographic drug ordering (dA <= dB) and swap concentrations accordingly.
    sw = df["dA"].astype(str) > df["dB"].astype(str)
    for a, b in [("dA", "dB"), ("cA", "cB")]:
        va, vb = df.loc[sw, a].copy(), df.loc[sw, b].copy()
        df.loc[sw, a], df.loc[sw, b] = vb, va
    report["rows_swapped"] = int(sw.sum())

    # Average available technical replicates per row (viability1..4) and scale by 100.
    for i in range(1, 5):
        df[f"v{i}"] = pd.to_numeric(df[f"v{i}"], errors="coerce")
    df["V_MEAN"] = df[["v1", "v2", "v3", "v4"]].mean(axis=1)
    df["PERCENTGROWTH"] = 100.0 * df["V_MEAN"]

    mu = pd.to_numeric(df["mu"], errors="coerce")
    report["target_definition"] = "100 * mean(available viability1..4)"
    report["mu_norm_note"] = "Target uses mean viability directly rather than mu/muMax"
    report["mu_range"] = [float(np.nanmin(mu)), float(np.nanmax(mu))]
    report["v_mean_range"] = [float(np.nanmin(df["V_MEAN"])), float(np.nanmax(df["V_MEAN"]))]
    report["y_range"] = [float(np.nanmin(df["PERCENTGROWTH"])),
                         float(np.nanmax(df["PERCENTGROWTH"]))]

    # O'Neil source concentrations are already reported in uM ("drugA Conc (uM)").
    for c in ["cA", "cB"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["CONC1"], df["CONC2"] = df["cA"], df["cB"]
    lo, hi = float(df["CONC1"].min()), float(df["CONC1"].max())
    if not (1e-5 <= lo and hi <= 1e5):
        raise ValueError(f"Unexpected O'Neil concentration range [{lo:.3e}, {hi:.3e}] uM")
    report["conc_unit"] = "uM (as reported in source)"
    df["CONC1_LOG"] = np.log10(df["CONC1"].where(df["CONC1"] > 0))
    df["CONC2_LOG"] = np.log10(df["CONC2"].where(df["CONC2"] > 0))
    report["conc_log_range"] = [float(df["CONC1_LOG"].min()),
                                float(df["CONC1_LOG"].max())]

    # Pool exact concentration duplicates within (cell, drugA, drugB, conc1, conc2).
    n_before = len(df)
    key = ["cell", "dA", "dB", "CONC1", "CONC2"]
    agg = {c: "mean" for c in ["PERCENTGROWTH", "CONC1_LOG", "CONC2_LOG"]}
    for c in ["v1", "v2", "v3", "v4", "mu", "X0"]:
        agg[c] = "mean"
    df = df.groupby(key, as_index=False).agg(agg)
    report["rows_deduped"] = n_before
    report["rows_after_dedup"] = len(df)
    report["dup_removed"] = n_before - len(df)
    log(f"  Deduplicated: {n_before:,} -> {len(df):,} (removed {n_before - len(df):,})")

    df = df.rename(columns={"cell": "CELLNAME", "dA": "NSC1", "dB": "NSC2"})
    df["PAIR"] = df["NSC1"].astype(str) + "_" + df["NSC2"].astype(str)
    df["EXPECTEDGROWTH"] = np.nan
    df["SCORE"] = np.nan
    df["CONCINDEX1"] = np.nan
    df["CONCINDEX2"] = np.nan
    df["SCREENER"] = "ONEIL"
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-subset", default=None,
                    help="JSON listing O'Neil cell lines to retain")
    ap.add_argument("--n-genes", type=int, default=1000)
    ap.add_argument("--gene-panel", default=None,
                    help="Gene panel JSON path or 'variance' for top-variance fallback (default: configs/gene_panel_l1000.json)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--extra-smiles", default=None,
                    help="JSON with supplementary drug SMILES strings")
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--configs-dir", default=os.path.join(ROOT, "configs"))
    ap.add_argument("--skip-expr", action="store_true")
    args = ap.parse_args()

    if args.gene_panel is None:
        args.gene_panel = os.path.join(ROOT, "configs", "gene_panel_l1000.json")

    cfg = args.config or os.path.basename(os.path.normpath(args.out))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log(f"Config={cfg} cell_source=depmap")

    maps = json.load(open(os.path.join(args.configs_dir, "cell_maps.json"), encoding="utf-8"))
    ol_info = maps["oneil"]

    subset = None
    if args.cell_subset:
        subset = set(json.load(open(args.cell_subset, encoding="utf-8"))["cells"])
        log(f"Cell subset: {len(subset)} cell lines")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg, "dataset": "O'Neil 2016 (AACR supplementary)",
        "cell_source": "DepMap 25Q3 RNA-seq (log2(TPM+1))",
        "target_column": "PERCENTGROWTH (100 * mean(available viability1..4))",
        "batch_policy": "legacy_exact_concentration_pooling",
        "split_ratio": "60/20/20",
        "note_expectedgrowth": "NA (single-agent expected growth not supplied in source workbook)",
        "args": vars(args),
    }

    df = load_oneil(os.path.join(args.raw_dir, "oneil", "oneil_raw.csv"), report)
    df = add_dose_position(df)

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

        p = os.path.join(args.raw_dir, "depmap", DEPMAP_CSV)
        log(f"Reading DepMap: {DEPMAP_CSV}")
        labels, X, genes = load_depmap(
            p, args.n_genes,
            pc_entrez if args.gene_panel == "variance" else None,
            pc_syms if args.gene_panel == "variance" else None,
            panel=panel, amap=amap)
        log(f"  Expression shape={X.shape} ({len(labels)} models x {len(genes)} genes)")
        lab_idx = {l: i for i, l in enumerate(labels)}
        report["expression_shape"] = [int(x) for x in X.shape]
        report["genes_top5"] = list(genes[:5])
    else:
        labels, X, genes, lab_idx = [], None, [], {}

    ol_cells = sorted(df["CELLNAME"].unique(), key=norm)
    if subset is not None:
        ol_cells = [c for c in ol_cells if c in subset]
    mapping, unmapped = {}, []
    for name in ol_cells:
        ach = ol_info.get(name, {}).get("ach", "")
        if ach and ach in lab_idx:
            mapping[name] = ach
        else:
            unmapped.append(name)
    report["cell_lines_considered"] = len(ol_cells)
    report["cell_lines_mapped"] = len(mapping)
    report["cell_lines_unmapped"] = unmapped
    log(f"  Cell mapping: {len(mapping)}/{len(ol_cells)}"
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
              "SCREENER", "PAIR", "NSC1", "NSC2", "CELLNAME"]].copy()
    panel_of = build_panel_map(os.path.join(args.raw_dir, "drugcomb",
                                            "DrugComb_cell_line_identifiers.xlsx"))
    out["PANEL"] = [panel_of.get(mapping.get(c, ""), "") for c in out["CELLNAME"]]
    for c in ["PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE", "CONC1", "CONC2",
              "CONC1_LOG", "CONC2_LOG", "CONC1_LOG_POS", "CONC2_LOG_POS"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
    for c in ["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX"]:
        out[c] = out[c].astype(np.int32)
    out = out[["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX", "CONC1", "CONC2",
               "CONC1_LOG", "CONC2_LOG", "CONC1_LOG_POS", "CONC2_LOG_POS",
               "PERCENTGROWTH", "EXPECTEDGROWTH", "SCORE",
               "SCREENER", "PANEL", "PAIR", "NSC1", "NSC2", "CELLNAME"]]
    out.to_parquet(os.path.join(args.out, "pairs.parquet"), index=False, compression="zstd")
    report["main_table_rows"] = len(out)

    json.dump({"cell_to_idx": cell2idx,
               "idx_to_cell": {str(i): c for c, i in cell2idx.items()},
               "cell_key": mapping},
              open(os.path.join(args.out, "cell_dict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"drug_to_idx": {str(n): i for n, i in d2i.items()},
               "idx_to_drug": {str(i): str(n) for n, i in d2i.items()}},
              open(os.path.join(args.out, "drug_dict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    smi_map = load_drug_smiles(os.path.join(args.raw_dir, "drugcomb",
                                            "DrugComb_drug_identifiers.xlsx"))
    if args.extra_smiles and os.path.exists(args.extra_smiles):
        extra = json.load(open(args.extra_smiles, encoding="utf-8"))
        for k, v in extra.items():
            smi_map.setdefault(norm(k), v)
        log(f"  Loaded supplementary SMILES for {list(extra.keys())}")
    smiles = [smi_map.get(norm(d)) for d in drugs]
    missing = [d for d, s in zip(drugs, smiles) if not s]
    report["drugs_with_smiles"] = len(drugs) - len(missing)
    report["drugs_missing_smiles"] = missing
    json.dump({"order": [str(d) for d in drugs], "smiles": smiles},
              open(os.path.join(args.out, "drug_smiles.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log(f"  Drug SMILES: {len(drugs) - len(missing)}/{len(drugs)}"
        + (f" (missing={missing})" if missing else ""))

    report["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(jsonable(report), open(os.path.join(args.out, "preprocess_report.json"), "w",
                                     encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

    log(f"Finished [{cfg}] rows={report['main_table_rows']:,} cells={report['n_cells']} "
        f"drugs={report['n_drugs']} pairs={report['n_pairs']} elapsed={report['elapsed_sec']}s")


if __name__ == "__main__":
    main()
