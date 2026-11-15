# -*- coding: utf-8 -*-
"""
生成四种分子表征（缺口1：分子表征谱系）
==========================================

产出 <data>/drug_reprs.npz，包含：
  onehot    (104, 104)     恒等矩阵，无化学信息（下界）
  rdkit     (104, D)       RDKit 理化描述符（标准化）
  morgan    (104, 2048)    Morgan 指纹
  chemberta (104, 768)     ChemBERTa 分子语言模型嵌入（均值池化）
  nsc       (104,)         药物顺序

用法：
    python make_drug_reprs.py --data data/data_v2 --chemberta <模型目录>
"""
import argparse
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def log(m):
    print(m, flush=True)


def parse_smiles(smi, Chem):
    """解析 SMILES。金属配合物默认价键规则会拒绝，降级处理。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return mol, "normal"
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None, "failed"
    try:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        return mol, "relaxed"
    except Exception:
        return mol, "unsanitized"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--chemberta", default=None, help="ChemBERTa 模型目录")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")

    d = json.load(open(os.path.join(args.data, "drug_smiles.json"), encoding="utf-8"))
    order, smiles = d["order"], d["smiles"]
    n = len(order)
    log(f"药物 {n} 个")

    mols, hows = [], []
    for s in smiles:
        m, h = parse_smiles(s if isinstance(s, str) else "", Chem)
        mols.append(m)
        hows.append(h)
    from collections import Counter
    log(f"SMILES 解析: {dict(Counter(hows))}")

    out = {"nsc": np.array(order)}

    # ---- 1. one-hot ----
    out["onehot"] = np.eye(n, dtype=np.float32)
    log(f"onehot    {out['onehot'].shape}")

    # ---- 2. RDKit 描述符 ----
    names = [x[0] for x in Descriptors.descList]
    # 逐分子算全部描述符
    raw = np.full((n, len(names)), np.nan, dtype=np.float64)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        for j, (nm, fn) in enumerate(Descriptors.descList):
            try:
                v = fn(mol)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    raw[i, j] = v
            except Exception:
                pass
    # 保留在 ≥95% 分子上都有效的描述符
    valid = np.isfinite(raw).mean(axis=0) >= 0.95
    keep_idx = np.flatnonzero(valid)
    desc = raw[:, keep_idx]
    # 缺失值用列中位填补
    for j in range(desc.shape[1]):
        col = desc[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            desc[bad, j] = np.nanmedian(col)
    # 去掉常数与极端量纲：截尾 + 标准化
    desc = np.clip(desc, np.nanpercentile(desc, 1, axis=0), np.nanpercentile(desc, 99, axis=0))
    df_mean, df_std = desc.mean(0), desc.std(0)
    df_std[df_std == 0] = 1.0
    desc = ((desc - df_mean) / df_std).astype(np.float32)
    out["rdkit"] = desc
    out["rdkit_names"] = np.array([names[i] for i in keep_idx])
    log(f"rdkit     {desc.shape}   （保留 {len(keep_idx)}/{len(names)} 个描述符）")

    # ---- 3. Morgan ----
    mor = np.zeros((n, 2048), dtype=np.float32)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            mor[i] = np.array(fp, dtype=np.float32)
        except Exception:
            pass
    out["morgan"] = mor
    log(f"morgan    {mor.shape}   （非零位均值 {mor.sum(1).mean():.1f}）")

    # ---- 4. ChemBERTa ----
    if args.chemberta and os.path.isdir(args.chemberta):
        log(f"加载 ChemBERTa: {args.chemberta}")
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(args.chemberta)
        mdl = AutoModel.from_pretrained(args.chemberta).to(args.device).eval()
        embs = np.zeros((n, mdl.config.hidden_size), dtype=np.float32)
        with torch.no_grad():
            for i, s in enumerate(smiles):
                if not isinstance(s, str) or not s:
                    continue
                enc = tok(s, return_tensors="pt", padding=True, truncation=True,
                          max_length=128).to(args.device)
                o = mdl(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                embs[i] = ((o * mask).sum(1) / mask.sum(1)).cpu().numpy()[0]
        out["chemberta"] = embs
        log(f"chemberta {embs.shape}")
        del mdl
        import torch as _t
        _t.cuda.empty_cache()
    else:
        log("跳过 ChemBERTa（未指定模型目录）")

    p = os.path.join(args.data, "drug_reprs.npz")
    np.savez_compressed(p, **out)
    log(f"已写出 {p}")
    log("键: " + ", ".join(f"{k}{out[k].shape}" for k in
                          ["onehot", "rdkit", "morgan", "chemberta"] if k in out))


if __name__ == "__main__":
    main()
