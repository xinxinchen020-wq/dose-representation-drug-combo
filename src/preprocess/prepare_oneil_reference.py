# -*- coding: utf-8 -*-
"""
O'Neil 2016 原有预处理（跨批次按精确浓度聚合）
==============================

来源：AACR 补充材料 `15357163mct150843-sup-156849_1_supp_1_w2lrww.xls`
     （实为 xlsx；O'Neil et al. 2016, Mol Cancer Ther 15(6):1155-62）
     已转存为 raw/oneil/oneil_raw.csv

原始列：BatchID, cell_line, drugA_name, drugA Conc (µM), drugB_name,
        drugB Conc (µM), combination_name, viability1..4, mu/muMax, X/X0

映射到统一契约：
    目标量   PERCENTGROWTH = 100 × mean(viability1..4)
    浓度     CONC1/CONC2   = 原始 µM 浓度（不换算）
    实体键   NSC1/NSC2     = 药物名（字符串）
    表达谱    DepMap 25Q3（与 C3 同源同基因集）

目标定义
    使用每行可用 viability1..4 的均值乘100，不使用 mu/muMax。
    O'Neil 的相对对照存活读数与ALMANAC含加药前基线的生长百分比定义不同，
    不能因为列名相同就把两个数据集的响应直接对齐。

2026-10-12批次核查
    本脚本保留旧版本行为，跨批次直接按精确浓度聚合可能产生额外剂量档。
    剂量重复实验使用 prepare_oneil_batch_checked.py 生成独立批次核对版本。

⚠️ 与 ALMANAC 的差异（必须在论文中说明）：
    1. 无单药数据 → EXPECTEDGROWTH / SCORE 置 NaN
    2. 原始每个单批次是4×4网格，旧版本合并批次后的网格可能更大
    3. 标度不完全相同（不同实验体系、不同对照）→ 不做跨配置绝对值比较

用法：
    python prep_oneil.py --out data/processed/oneil
    python prep_oneil.py --out data/processed/oneil --extra-smiles data/configs/extra_smiles.json
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from src.preprocess.prep_common import (add_dose_position, jsonable, load_alias_map, load_depmap,
                         load_drug_smiles, load_gene_panel, load_hgnc, norm)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

DEPMAP_CSV = "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
HGNC_TSV = "hgnc_complete_set.txt"

RAW_COLS = ["cell", "dA", "cA", "dB", "cB", "combo",
            "v1", "v2", "v3", "v4", "mu", "X0"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_panel_map(xlsx_path):
    """DrugComb 细胞系表 → {ACH: 组织}（取自 ccle_name 的后缀，如 A2058_SKIN）"""
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
    """读取并规范化 O'Neil 原始表"""
    log(f"读取 {os.path.basename(csv_path)} ...")
    df = pd.read_csv(csv_path, index_col=0)
    if len(df.columns) != len(RAW_COLS):
        raise ValueError(f"列数不符：期望 {len(RAW_COLS)}，实际 {len(df.columns)}")
    df.columns = RAW_COLS
    log(f"  {len(df):,} 行")
    report["rows_raw"] = len(df)

    # --- 规范药物顺序（dA < dB），同步交换浓度
    sw = df["dA"].astype(str) > df["dB"].astype(str)
    for a, b in [("dA", "dB"), ("cA", "cB")]:
        va, vb = df.loc[sw, a].copy(), df.loc[sw, b].copy()
        df.loc[sw, a], df.loc[sw, b] = vb, va
    report["rows_swapped"] = int(sw.sum())

    # --- 目标量
    # 每行最多四个技术重复；缺失读数不当作0，平均其余可用读数。
    # 本响应与ALMANAC含加药前基线的生长百分比定义不同。
    for i in range(1, 5):
        df[f"v{i}"] = pd.to_numeric(df[f"v{i}"], errors="coerce")
    df["V_MEAN"] = df[["v1", "v2", "v3", "v4"]].mean(axis=1)
    df["PERCENTGROWTH"] = 100.0 * df["V_MEAN"]

    mu = pd.to_numeric(df["mu"], errors="coerce")
    report["target_definition"] = "100 × mean(available viability1..4) （相对对照的存活读数）"
    report["mu_norm_note"] = "当前目标直接取自viability读数，原始mu/muMax列未用作目标"
    report["mu_range"] = [float(np.nanmin(mu)), float(np.nanmax(mu))]
    report["v_mean_range"] = [float(np.nanmin(df["V_MEAN"])), float(np.nanmax(df["V_MEAN"]))]
    report["y_range"] = [float(np.nanmin(df["PERCENTGROWTH"])),
                         float(np.nanmax(df["PERCENTGROWTH"]))]

    # --- 浓度
    # ★ O'Neil 原始补充材料的列名显式声明单位为 µM（"drugA Conc (µM)"），无需换算。
    #   与 ALMANAC 对齐：ALMANAC 原始是 M，在 prep_almanac.py 里已 ×1e6 转成 µM。
    for c in ["cA", "cB"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["CONC1"], df["CONC2"] = df["cA"], df["cB"]
    lo, hi = float(df["CONC1"].min()), float(df["CONC1"].max())
    if not (1e-5 <= lo and hi <= 1e5):
        raise ValueError(f"O'Neil 浓度值域 [{lo:.3e}, {hi:.3e}] 不像 µM，请核查单位")
    report["conc_unit"] = "µM（原始即 µM，未换算）"
    df["CONC1_LOG"] = np.log10(df["CONC1"].where(df["CONC1"] > 0))
    df["CONC2_LOG"] = np.log10(df["CONC2"].where(df["CONC2"] > 0))
    report["conc_log_range"] = [float(df["CONC1_LOG"].min()),
                                float(df["CONC1_LOG"].max())]

    # --- 去重：(细胞系, 药对, 浓度1, 浓度2) 唯一
    #     315 个块有 32 行（双份测量），取均值避免同一条件出现两个目标值
    n_before = len(df)
    key = ["cell", "dA", "dB", "CONC1", "CONC2"]
    agg = {c: "mean" for c in ["PERCENTGROWTH", "CONC1_LOG", "CONC2_LOG"]}
    for c in ["v1", "v2", "v3", "v4", "mu", "X0"]:
        agg[c] = "mean"
    df = df.groupby(key, as_index=False).agg(agg)
    report["rows_deduped"] = n_before
    report["rows_after_dedup"] = len(df)
    report["dup_removed"] = n_before - len(df)
    log(f"  去重: {n_before:,} -> {len(df):,}（移除 {n_before-len(df):,}）")

    df = df.rename(columns={"cell": "CELLNAME", "dA": "NSC1", "dB": "NSC2"})
    df["PAIR"] = df["NSC1"].astype(str) + "_" + df["NSC2"].astype(str)
    df["EXPECTEDGROWTH"] = np.nan
    df["SCORE"] = np.nan
    df["CONCINDEX1"] = np.nan
    df["CONCINDEX2"] = np.nan
    df["SCREENER"] = "ONEIL"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-subset", default=None)
    ap.add_argument("--n-genes", type=int, default=1000)
    ap.add_argument("--gene-panel", default=None,
                    help="固定基因面板 JSON 路径；传 'variance' 回退到方差 top-N。"
                         "默认 data/configs/gene_panel_l1000.json")
    ap.add_argument("--config", default=None)
    ap.add_argument("--extra-smiles", default=None,
                    help="补充 SMILES 的 JSON（如 PubChem 补的奥沙利铂）")
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--configs-dir", default=os.path.join(ROOT, "data", "configs"))
    ap.add_argument("--skip-expr", action="store_true")
    args = ap.parse_args()

    if args.gene_panel is None:
        args.gene_panel = os.path.join(ROOT, "data", "configs", "gene_panel_l1000.json")

    cfg = args.config or os.path.basename(os.path.normpath(args.out))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log(f"配置 {cfg}   细胞系表征 = depmap")

    maps = json.load(open(os.path.join(args.configs_dir, "cell_maps.json"), encoding="utf-8"))
    ol_info = maps["oneil"]

    subset = None
    if args.cell_subset:
        subset = set(json.load(open(args.cell_subset, encoding="utf-8"))["cells"])
        log(f"细胞系子集: {len(subset)} 系")

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": cfg, "dataset": "O'Neil 2016 (AACR supplementary)",
              "cell_source": "DepMap 25Q3 RNA-seq (log2(TPM+1))",
              "target_column": "PERCENTGROWTH (= 100 × mean(available viability1..4))",
              "batch_policy": "legacy_exact_concentration_pooling",
              "split_ratio": "60/20/20",
              "note_expectedgrowth": "NA —— 原始数据无单药响应，无法算加性期望",
              "args": vars(args)}

    df = load_oneil(os.path.join(args.raw_dir, "oneil", "oneil_raw.csv"), report)
    df = add_dose_position(df)

    # ---------- 表达谱
    if not args.skip_expr:
        hgnc = os.path.join(args.raw_dir, "gene_lists", HGNC_TSV)
        panel = amap = None
        if args.gene_panel != "variance":
            panel, _pj = load_gene_panel(args.gene_panel)
            amap = load_alias_map(hgnc)
            log(f"基因面板 {_pj['panel']}: {len(panel)} 个基因")
            report["gene_panel"] = {"name": _pj["panel"], "n_genes": len(panel),
                                    "source": _pj.get("source", ""),
                                    "file": args.gene_panel}
        else:
            pc_entrez, pc_syms = load_hgnc(hgnc)
            report["protein_coding_entrez"] = len(pc_entrez)
            report["gene_panel"] = {"name": "variance_top", "n_genes": args.n_genes}

        p = os.path.join(args.raw_dir, "depmap", DEPMAP_CSV)
        log(f"读取 DepMap: {DEPMAP_CSV}")
        labels, X, genes = load_depmap(
            p, args.n_genes,
            pc_entrez if args.gene_panel == "variance" else None,
            pc_syms if args.gene_panel == "variance" else None,
            panel=panel, amap=amap)
        log(f"  表达谱 {X.shape}（{len(labels)} 系 × {len(genes)} 基因）")
        lab_idx = {l: i for i, l in enumerate(labels)}
        report["expression_shape"] = [int(x) for x in X.shape]
        report["genes_top5"] = list(genes[:5])
    else:
        labels, X, genes, lab_idx = [], None, [], {}

    # ---------- 细胞系筛选
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
    log(f"  细胞系映射: {len(mapping)}/{len(ol_cells)}" +
        (f"，未映射 {unmapped}" if unmapped else ""))

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

    # ---------- 表达谱重排
    if X is not None:
        order = [lab_idx[mapping[c]] for c in cells_used]
        np.save(os.path.join(args.out, "cell_expression.npy"), X[order])
        json.dump({"cell_names": cells_used, "genes": genes},
                  open(os.path.join(args.out, "cell_expression_names.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
        log(f"  表达谱重排 {X[order].shape}")

    # ---------- 主表
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

    # ---------- 字典
    json.dump({"cell_to_idx": cell2idx,
               "idx_to_cell": {str(i): c for c, i in cell2idx.items()},
               "cell_key": mapping},
              open(os.path.join(args.out, "cell_dict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"drug_to_idx": {str(n): i for n, i in d2i.items()},
               "idx_to_drug": {str(i): str(n) for n, i in d2i.items()}},
              open(os.path.join(args.out, "drug_dict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------- SMILES
    smi_map = load_drug_smiles(os.path.join(args.raw_dir, "drugcomb",
                                            "DrugComb_drug_identifiers.xlsx"))
    if args.extra_smiles and os.path.exists(args.extra_smiles):
        extra = json.load(open(args.extra_smiles, encoding="utf-8"))
        for k, v in extra.items():
            smi_map.setdefault(norm(k), v)
        log(f"  载入补充 SMILES: {list(extra.keys())}")
    smiles = [smi_map.get(norm(d)) for d in drugs]
    missing = [d for d, s in zip(drugs, smiles) if not s]
    report["drugs_with_smiles"] = len(drugs) - len(missing)
    report["drugs_missing_smiles"] = missing
    json.dump({"order": [str(d) for d in drugs], "smiles": smiles},
              open(os.path.join(args.out, "drug_smiles.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log(f"  药物 SMILES: {len(drugs)-len(missing)}/{len(drugs)}" +
        (f"，缺 {missing}" if missing else ""))

    report["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(jsonable(report), open(os.path.join(args.out, "preprocess_report.json"), "w",
                                     encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

    log(f"完成 [{cfg}]  行={report['main_table_rows']:,}  细胞系={report['n_cells']}  "
        f"药={report['n_drugs']}  药对={report['n_pairs']}  耗时={report['elapsed_sec']}s")


if __name__ == "__main__":
    main()
