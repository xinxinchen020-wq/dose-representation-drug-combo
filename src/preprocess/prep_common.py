# -*- coding: utf-8 -*-
"""
数据预处理共用模块
====================

服务于四个数据配置：
    C1 almanac_cm59   NCI-ALMANAC + CellMiner, 59 系
    C2 almanac_cm47   NCI-ALMANAC + CellMiner, 47 系
    C3 almanac_dm47   NCI-ALMANAC + DepMap,   47 系
    C4 oneil          O'Neil 2016 + DepMap,   35 系

统一约定（所有配置必须一致，否则跨配置比较不成立）：

  1. **表达谱值都在 log 空间**，不要再取 log
       CellMiner : log2(FPKM + 1)
       DepMap    : log2(TPM  + 1)
  2. **基因筛选规则**：HGNC 蛋白编码基因 → 在【该资源的全部细胞系】上
     取方差最大的 N 个。注意是「资源的全部细胞系」而非「配置的子集」，
     这样 C1/C2 共享同一基因集、C3/C4 共享同一基因集，
     使 C1-vs-C2（只换子集）与 C3-vs-C4（只换数据集）成为干净对照。
  3. **细胞系映射**统一走 DrugComb 细胞系标识符表
     （含 depmap_id / ccle_name / cosmic_id / cell_model_passport_id）。
"""
import json
import re

import numpy as np
import pandas as pd

CLEAN_RE = re.compile(r"[\x00-\x1f\x7f]")

# ------------------------------------------------------------------ 名称归一化


def norm(s):
    """细胞系名归一化：去控制字符 → 去非字母数字 → 小写"""
    if not isinstance(s, str):
        return ""
    s = CLEAN_RE.sub("", s).strip()
    return re.sub(r"[^A-Za-z0-9]", "", s).lower()


def variants(n):
    """一个细胞系名的全部等价写法（去后缀、去括号、去斜杠后半段）"""
    n = str(n)
    out = {norm(n)}
    for suf in ["/ATCC", "-ATCC", "(TB)", "211H"]:
        if n.upper().endswith(suf.upper()):
            out.add(norm(n[: -len(suf)]))
    if "/" in n:
        out.add(norm(n.split("/")[0]))
    if "(" in n:
        out.add(norm(re.sub(r"\(.*?\)", "", n)))
    return {x for x in out if x}


# ------------------------------------------------------------------ HGNC


def load_hgnc(tsv_path):
    """返回 (蛋白编码基因的 Entrez 集合, 蛋白编码基因的符号集合)"""
    d = pd.read_csv(tsv_path, sep="\t", low_memory=False,
                    usecols=["hgnc_id", "symbol", "locus_group", "entrez_id"])
    pc = d[d["locus_group"] == "protein-coding gene"]
    entrez = set(pd.to_numeric(pc["entrez_id"], errors="coerce").dropna().astype(np.int64))
    syms = set(pc["symbol"].dropna().astype(str).str.upper())
    return entrez, syms


def _variance_top(X, genes, n_genes):
    """在 X (细胞系 × 基因) 上取方差最大的 n_genes 个基因

    ⚠️ 仅作回退路径。默认应使用固定面板（见下），否则 C1/C2 与 C3/C4 的基因集不同，
       C2-vs-C3 的「只差表达谱来源」不成立。
    """
    var = np.var(X, axis=0)
    top = np.argsort(var)[::-1][:n_genes]
    return X[:, top], [genes[i] for i in top]


# ------------------------------------------------------------------ 基因符号规范化


def load_alias_map(hgnc_tsv):
    """{任意符号（含别名/旧名，大写）: 现行 HGNC 符号（大写）}

    CellMiner 与 DepMap 的符号版本不同（AARS vs AARS1、ADCK3 vs COQ8A …），
    必须先规范化到同一套符号才能比较。
    """
    h = pd.read_csv(hgnc_tsv, sep="\t", low_memory=False,
                    usecols=["symbol", "alias_symbol", "prev_symbol"])
    amap = {}
    for _, r in h.iterrows():
        cur = str(r["symbol"]).strip().upper()
        if not cur:
            continue
        amap.setdefault(cur, cur)
        for col in ("alias_symbol", "prev_symbol"):
            v = r.get(col)
            if isinstance(v, str):
                for a in re.split(r"[|,]", v):
                    a = a.strip().upper()
                    if a:
                        amap.setdefault(a, cur)
    return amap


def canon_symbol(sym, amap):
    s = str(sym).strip().upper()
    return amap.get(s, s)


def load_gene_panel(json_path):
    """读取固定基因面板 → (基因列表, 完整 JSON)"""
    d = json.load(open(json_path, encoding="utf-8"))
    return list(d["genes"]), d


def _select_panel(X, genes, panel, amap, label):
    """按固定面板选基因并重排到面板顺序。

    返回 (X_sel, genes_sel, missing)。genes_sel 的顺序与 panel 一致，
    因此同一面板在所有配置上产出完全相同的基因列表。
    """
    idx = {}
    for i, g in enumerate(genes):
        c = canon_symbol(g, amap)
        if c not in idx:
            idx[c] = i
    cols, keep, miss = [], [], []
    for g in panel:
        if g in idx:
            cols.append(idx[g])
            keep.append(g)
        else:
            miss.append(g)
    if miss:
        print(f"  [{label}] 面板中有 {len(miss)} 个基因在本资源中缺失: {miss[:10]}"
              f"{' ...' if len(miss) > 10 else ''}")
    return X[:, cols], keep, miss


# ------------------------------------------------------------------ CellMiner


def load_cellminer(xls_path, n_genes, pc_entrez=None, pc_symbols=None,
                   panel=None, amap=None):
    """读 CellMiner 表达谱（基因行 × 细胞系列）→ (细胞系名, X[cells×genes], 基因名)

    表头在第 11 行（0-based 10），前 6 列是基因元信息。
    细胞系列名形如 'BR:MCF7'，剥掉组织前缀。

    panel 非空时按固定面板选基因（四配置共用，推荐）；否则回退到方差 top-N。
    """
    df = pd.read_excel(xls_path, header=10)
    df = df.dropna(how="all")
    meta = list(df.columns[:6])
    raw_cells = [str(c).strip() for c in df.columns[6:]]
    cells = [c.split(":", 1)[1] if ":" in c else c for c in raw_cells]

    genes = df[meta[0]].astype(str).values
    entrez = pd.to_numeric(df[meta[1]], errors="coerce")
    X = df.iloc[:, 6:].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    X = X.T                                       # (基因, 细胞系) -> (细胞系, 基因)

    X = np.nan_to_num(X, nan=0.0)
    if panel is not None:
        X, genes, _ = _select_panel(X, genes, panel, amap, "CellMiner")
        return cells, X, genes

    e = entrez.fillna(-1).astype(np.int64).values
    keep = np.array([(ei in pc_entrez) or (str(g).upper() in pc_symbols)
                     for ei, g in zip(e, genes)])
    X, genes = X[:, keep], [g for g, k in zip(genes, keep) if k]
    X, genes = _variance_top(X, genes, n_genes)
    return cells, X, genes


# ------------------------------------------------------------------ DepMap


DEPMAP_META = ["SequencingID", "ModelID", "IsDefaultEntryForModel",
               "ModelConditionID", "IsDefaultEntryForMC"]


def load_depmap(csv_path, n_genes, pc_entrez=None, pc_symbols=None,
                panel=None, amap=None):
    """读 DepMap OmicsExpression → (ACH 列表, X[models×genes], 基因名)

    列名格式 'SYMBOL (ENTREZ)'；同一 ModelID 可能有多条测序记录，
    取 IsDefaultEntryForModel == 'Yes' 的那条。

    panel 非空时按固定面板选基因（四配置共用，推荐）；否则回退到方差 top-N。
    """
    df = pd.read_csv(csv_path, low_memory=False)
    # 第一列是行号
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]

    meta = [c for c in DEPMAP_META if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta]

    sym, ent, keep_idx = [], [], []
    for i, c in enumerate(gene_cols):
        m = re.match(r"^(.*?)\s*\((\d+)\)$", str(c))
        if not m:
            continue
        s, e = m.group(1), int(m.group(2))
        sym.append(s)
        ent.append(e)
        keep_idx.append(i)

    X = df[gene_cols].values.astype(np.float32)[:, keep_idx]
    X = np.nan_to_num(X, nan=0.0)

    ids = df["ModelID"].astype(str).values
    isdef = df["IsDefaultEntryForModel"].astype(str).str.lower().values
    sel = isdef == "yes"
    if sel.sum() == 0:
        sel = np.ones(len(df), dtype=bool)
    ach = ids[sel]
    X = X[sel]

    # 同一 ModelID 若仍重复，取第一条
    _, uniq = np.unique(ach, return_index=True)
    uniq = np.sort(uniq)
    ach, X = ach[uniq], X[uniq]

    if panel is not None:
        X, sym, _ = _select_panel(X, sym, panel, amap, "DepMap")
        return list(ach), X, sym

    keep = np.array([(e in pc_entrez) or (s.upper() in pc_symbols)
                     for s, e in zip(sym, ent)])
    X, sym = X[:, keep], [s for s, k in zip(sym, keep) if k]
    X, sym = _variance_top(X, sym, n_genes)
    return list(ach), X, sym


# ------------------------------------------------------------------ 细胞系映射


def load_drugcomb_cells(xlsx_path):
    """DrugComb 细胞系标识符表 → {归一化名: {全部候选 ACH}}"""
    dc = pd.read_excel(xlsx_path, sheet_name="cell_line")
    name2achs = {}
    for _, r in dc.iterrows():
        dm = r.get("depmap_id")
        if pd.isna(dm) or not str(dm).startswith("ACH"):
            continue
        keys = variants(r["name"])
        if isinstance(r.get("synonyms"), str):
            for s in r["synonyms"].split(";"):
                keys |= variants(s)
        if isinstance(r.get("ccle_name"), str):
            keys |= variants(r["ccle_name"])
        for k in keys:
            name2achs.setdefault(k, set()).add(str(dm))
    return name2achs


def resolve_ach(cell, name2achs, depmap_ids):
    """细胞系名 → (命中的 ACH 或 '', 全部候选 ACH)"""
    cands = set()
    for v in variants(cell):
        cands |= name2achs.get(v, set())
    hit = sorted(a for a in cands if a in depmap_ids)
    return (hit[0] if hit else ""), sorted(cands)


def load_drug_smiles(xlsx_path):
    """DrugComb 药物标识符表 → {归一化药名: SMILES}"""
    dr = pd.read_excel(xlsx_path, sheet_name="drug")
    out = {}
    for _, r in dr.iterrows():
        smi = r.get("smiles")
        if not isinstance(smi, str) or not smi.strip():
            continue
        keys = {norm(r["dname"])}
        if isinstance(r.get("synonyms"), str):
            keys |= {norm(s) for s in re.split(r"[;,]", r["synonyms"]) if s.strip()}
        for k in keys:
            out.setdefault(k, smi.strip())
    return out


# ------------------------------------------------------------------ 杂项


def add_dose_position(df, c1="CONC1_LOG", c2="CONC2_LOG", pair="PAIR", cell="CELLNAME"):
    """在 (药对, 细胞系) 块内把 log 浓度归一到 [0,1] 位置

    与 ALMANAC 的 CONC*_LOG_POS 定义一致：块内 min→0, max→1。
    剂量插值/外推协议判定网格位置时用它。
    """
    g = df.groupby([pair, cell])
    for c in (c1, c2):
        lo = g[c].transform("min")
        hi = g[c].transform("max")
        rng = (hi - lo).replace(0, np.nan)
        df[c + "_POS"] = ((df[c] - lo) / rng).fillna(0.0)
    return df


def jsonable(o):
    """把 numpy 类型转成可 JSON 序列化的 Python 类型"""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
