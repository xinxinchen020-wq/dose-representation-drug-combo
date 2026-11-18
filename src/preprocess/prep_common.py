"""Shared preprocessing helpers for gene harmonization, expression loading, and cell/drug mapping."""
import json
import re

import numpy as np
import pandas as pd

CLEAN_RE = re.compile(r"[\x00-\x1f\x7f]")


def norm(s):
    """Normalize a cell-line or drug identifier to lowercase alphanumeric characters."""
    if not isinstance(s, str):
        return ""
    s = CLEAN_RE.sub("", s).strip()
    return re.sub(r"[^A-Za-z0-9]", "", s).lower()


def variants(n):
    """Generate normalized spelling variants for a cell-line name."""
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


def load_hgnc(tsv_path):
    """Return (protein_coding_entrez_ids, protein_coding_symbols) from HGNC complete set."""
    d = pd.read_csv(tsv_path, sep="\t", low_memory=False,
                    usecols=["hgnc_id", "symbol", "locus_group", "entrez_id"])
    pc = d[d["locus_group"] == "protein-coding gene"]
    entrez = set(pd.to_numeric(pc["entrez_id"], errors="coerce").dropna().astype(np.int64))
    syms = set(pc["symbol"].dropna().astype(str).str.upper())
    return entrez, syms


def _variance_top(X, genes, n_genes):
    """Fallback selection of the top n_genes highest-variance genes across matrix X."""
    var = np.var(X, axis=0)
    top = np.argsort(var)[::-1][:n_genes]
    return X[:, top], [genes[i] for i in top]


def load_alias_map(hgnc_tsv):
    """Build {uppercase_alias_or_previous_symbol: current_hgnc_symbol} lookup map."""
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
    """Load fixed gene panel JSON and return (ordered_gene_list, metadata_dict)."""
    d = json.load(open(json_path, encoding="utf-8"))
    return list(d["genes"]), d


def _select_panel(X, genes, panel, amap, label):
    """Select and align columns to a fixed ordered gene panel."""
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
        print(f"  [{label}] missing {len(miss)} panel genes: {miss[:10]}"
              f"{' ...' if len(miss) > 10 else ''}")
    return X[:, cols], keep, miss


def load_cellminer(xls_path, n_genes, pc_entrez=None, pc_symbols=None,
                   panel=None, amap=None):
    """Load CellMiner log2(FPKM + 1) expression matrix -> (cells, X[cells x genes], genes)."""
    df = pd.read_excel(xls_path, header=10)
    df = df.dropna(how="all")
    meta = list(df.columns[:6])
    raw_cells = [str(c).strip() for c in df.columns[6:]]
    cells = [c.split(":", 1)[1] if ":" in c else c for c in raw_cells]

    genes = df[meta[0]].astype(str).values
    entrez = pd.to_numeric(df[meta[1]], errors="coerce")
    X = df.iloc[:, 6:].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    X = X.T  # (genes, cells) -> (cells, genes)

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


DEPMAP_META = ["SequencingID", "ModelID", "IsDefaultEntryForModel",
               "ModelConditionID", "IsDefaultEntryForMC"]


def load_depmap(csv_path, n_genes, pc_entrez=None, pc_symbols=None,
                panel=None, amap=None):
    """Load DepMap log2(TPM + 1) expression matrix -> (model_ids, X[models x genes], genes)."""
    df = pd.read_csv(csv_path, low_memory=False)
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


def load_drugcomb_cells(xlsx_path):
    """Load DrugComb cell-line sheet -> {normalized_name: set_of_candidate_depmap_ids}."""
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
    """Resolve cell-line name -> (matched_ach_or_empty, sorted_candidate_achs)."""
    cands = set()
    for v in variants(cell):
        cands |= name2achs.get(v, set())
    hit = sorted(a for a in cands if a in depmap_ids)
    return (hit[0] if hit else ""), sorted(cands)


def load_drug_smiles(xlsx_path):
    """Load DrugComb drug sheet -> {normalized_drug_name: SMILES}."""
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


def add_dose_position(df, c1="CONC1_LOG", c2="CONC2_LOG", pair="PAIR", cell="CELLNAME"):
    """Normalize log doses to [0, 1] within each (drug pair, cell line) block."""
    g = df.groupby([pair, cell])
    for c in (c1, c2):
        lo = g[c].transform("min")
        hi = g[c].transform("max")
        rng = (hi - lo).replace(0, np.nan)
        df[c + "_POS"] = ((df[c] - lo) / rng).fillna(0.0)
    return df


def jsonable(o):
    """Recursively convert NumPy scalars/containers to standard JSON-serializable Python types."""
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
