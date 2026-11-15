"""Training and grid holdout routines from the frozen benchmark implementation."""
import argparse
import hashlib
import json
import os
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 剂量表征

from src.models.dose_encoders import DOSE_ENCS, fit_dose_input_transform, module_hash
from src.models.fusion_networks import ComboModel

def grid_structure(df):
    """用实际、互异的剂量点判断曲面完整性，重复测量不改变网格。"""
    g = df.groupby(["PAIR", "CELLNAME"], sort=False)
    r1 = g["CONC1_LOG"].rank(method="dense")
    r2 = g["CONC2_LOG"].rank(method="dense")
    n1 = g["CONC1_LOG"].transform("nunique")
    n2 = g["CONC2_LOG"].transform("nunique")
    unique_point = ~df.duplicated(["PAIR", "CELLNAME", "CONC1_LOG", "CONC2_LOG"])
    n_points = unique_point.groupby([df["PAIR"], df["CELLNAME"]], sort=False).transform("sum")
    finite = np.isfinite(df[["CONC1_LOG", "CONC2_LOG"]].to_numpy()).all(axis=1)
    all_finite = pd.Series(finite, index=df.index).groupby(
        [df["PAIR"], df["CELLNAME"]], sort=False).transform("all")
    complete = ((n1 * n2 == n_points) & (n1 >= 3) & (n2 >= 3) & all_finite)
    return r1, r2, n1, n2, complete


def grid_center_mask(df):
    """标记每个 (药对, 细胞系) 完整剂量网格的「中心点」

    中心按【秩】定义：n 档时中心秩为 {lo, hi}，lo=(n+1)//2, hi=(n+2)//2

        n=3 → {2}      1 点      ALMANAC 3×3：训外圈 8 点 → 测中心 1 点
        n=4 → {2,3}    4 点      O'Neil  4×4：训外圈 12 点 → 测中心 4 点
        n=5 → {3}      1 点
        n=6 → {3,4}    4 点

    只对**完整网格**（n1×n2 == 互异剂量点数）且 min(n1,n2) >= 3 的单元有效；
    其余单元返回 False（不参与 INTERP 的测试集）。
    """
    r1, r2, n1, n2, complete = grid_structure(df)
    lo1, hi1 = (n1 + 1) // 2, (n1 + 2) // 2
    lo2, hi2 = (n2 + 1) // 2, (n2 + 2) // 2
    in1 = (r1 >= lo1) & (r1 <= hi1)
    in2 = (r2 >= lo2) & (r2 <= hi2)
    return (complete & in1 & in2).fillna(False).values


# ---------------------------------------------------------------- 数据划分

#: 需要按「折」留出实体的协议
FOLD_PROTOCOLS = ("LCO", "LODO", "LDO")
#: 需要完整剂量网格的协议
GRID_PROTOCOLS = ("INTERP", "EXTRAP", "EXTRAP2")
DOSE_SPLIT_VERSION = "known-curve-dose-v3"
#: 全部协议
PROTOCOLS = ("LTO", "LPO", "LCO", "LODO", "LDO", "INTERP", "EXTRAP", "EXTRAP2")


def _entity_keys(df):
    """药物的实体键（LODO / LDO 用）。ALMANAC 为 NSC 整数，O'Neil 为药物名字符串。"""
    return df["NSC1"].values, df["NSC2"].values


def _pool_split(pair_codes, m_pool, m_te, rng, val_frac=0.25):
    """在池子内按药对切 train/val，避免泄漏。

    val_frac 是**池内**比例。实体/网格协议的测试集由结构决定，
    池内取 25% 作验证集 → 若测试集占 20%，则总体约为 60/20/20。
    """
    pool_codes = pair_codes[m_pool]
    uniq = np.unique(pool_codes)
    rng.shuffle(uniq)
    n_va = max(1, int(val_frac * len(uniq)))
    va_u = set(uniq[:n_va].tolist())
    m_va = m_pool & np.isin(pair_codes, list(va_u))
    m_tr = m_pool & ~m_va
    return m_tr, m_va, m_te


def make_split(df, protocol, seed=42, kfold=5, fold=0, val_frac=0.25):
    """返回 (train_idx, val_idx, test_idx)。

    原子单位（避免同一单元跨集合泄漏）：
      LTO    单位 = (药对, 细胞系, 剂量1, 剂量2)   —— 剂量点级
      LPO    单位 = 药对
      LCO    单位 = 细胞系（整折留出）
      LODO   单位 = 药物（整折留出；测试对中至少一个未知）
      LDO    单位 = 药物（整折留出；测试对中两个都未知，跨池行两边都不用）
      INTERP 验证/测试 = 完整曲面中心点，同曲面外圈用于训练
      EXTRAP 验证/测试 = 完整曲面药物1最高档，同曲面低档用于训练
      EXTRAP2 验证/测试 = 对应药物2最高档

    实现用 numpy 编码 + np.unique(return_inverse)，避免在百万行上做字符串拼接。
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    idx = np.arange(n)

    pair_codes, _ = pd.factorize(df["PAIR"].values)
    cell_arr = df["CELL_IDX"].values
    c1 = np.nan_to_num(df["CONC1_LOG"].values, nan=-99.0)
    c2 = np.nan_to_num(df["CONC2_LOG"].values, nan=-99.0)

    if protocol == "LTO":
        keys = np.stack([pair_codes, cell_arr,
                         (c1 * 1000).round().astype(np.int64),
                         (c2 * 1000).round().astype(np.int64)], axis=1)
        _, unit = np.unique(keys, axis=0, return_inverse=True)
        uniq = np.unique(unit); rng.shuffle(uniq)
        n_tr = int(0.6 * len(uniq)); n_va = int(0.2 * len(uniq))
        m_tr = np.isin(unit, uniq[:n_tr]); m_va = np.isin(unit, uniq[n_tr:n_tr + n_va])
        return idx[m_tr], idx[m_va], idx[~(m_tr | m_va)]

    if protocol == "LPO":
        uniq = np.unique(pair_codes); rng.shuffle(uniq)
        n_tr = int(0.6 * len(uniq)); n_va = int(0.2 * len(uniq))
        m_tr = np.isin(pair_codes, uniq[:n_tr])
        m_va = np.isin(pair_codes, uniq[n_tr:n_tr + n_va])
        return idx[m_tr], idx[m_va], idx[~(m_tr | m_va)]

    if protocol in GRID_PROTOCOLS:
        if not 0 < val_frac < 1:
            raise ValueError("剂量留出的验证药对比例必须在0和1之间")
        if protocol == "INTERP":
            held = grid_center_mask(df)
        else:
            r1, r2, n1, n2, complete = grid_structure(df)
            at_high = (r1 == n1) if protocol == "EXTRAP" else (r2 == n2)
            held = (complete & at_high).fillna(False).to_numpy()
        eligible_pairs = np.unique(pair_codes[held])
        if len(eligible_pairs) < 2:
            raise ValueError("剂量留出至少需要两个具有完整曲面的药对，以分开验证与测试")
        rng.shuffle(eligible_pairs)
        n_va = min(len(eligible_pairs) - 1,
                   max(1, int(val_frac * len(eligible_pairs))))
        val_pairs = eligible_pairs[:n_va]
        in_val = np.isin(pair_codes, val_pairs)
        m_tr = ~held
        m_va = held & in_val
        m_te = held & ~in_val
        return idx[m_tr], idx[m_va], idx[m_te]

    if protocol in FOLD_PROTOCOLS:
        d1, d2 = _entity_keys(df)
        if protocol == "LCO":
            ents = np.unique(cell_arr).copy()
        else:
            ents = np.unique(np.concatenate([d1, d2])).copy()
        rng.shuffle(ents)
        folds = np.array_split(ents, kfold)
        held = folds[fold]

        if protocol == "LCO":
            m_te = np.isin(cell_arr, held)
            m_pool = ~m_te
        else:
            in_h1 = np.isin(d1, held); in_h2 = np.isin(d2, held)
            if protocol == "LODO":
                m_te = in_h1 | in_h2                 # 至少一个药未知
                m_pool = ~m_te                       # 池 = 两个药都已知
            else:  # LDO
                m_te = in_h1 & in_h2                 # 两个药都未知
                train_drugs = np.setdiff1d(ents, held)
                m_pool = (np.isin(d1, train_drugs) & np.isin(d2, train_drugs))
                # 跨池的行两边都不用 → 数据大幅削减（参考论文报 ~68%）
        m_tr, m_va, _ = _pool_split(pair_codes, m_pool, m_te, rng, val_frac)
        return idx[m_tr], idx[m_va], idx[m_te]

    raise NotImplementedError(f"protocol {protocol} 尚未实现")


# ---------------------------------------------------------------- 指标

def metrics(y, p):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    e = y - p
    out = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}
    if y.std() > 0 and p.std() > 0:
        out["pearson"] = float(np.corrcoef(y, p)[0, 1])
        out["spearman"] = float(np.corrcoef(np.argsort(np.argsort(y)),
                                            np.argsort(np.argsort(p)))[0, 1])
    ss_res = np.sum(e ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    out["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    out["n"] = int(len(y))
    return out


def count_params(dose_enc_id, arch="A1", drug_dim=2048, cell_dim=977,
                 hidden=512, dropout=0.2):
    """统计某种剂量编码的全模型参数量及其自身参数量"""
    m = ComboModel(drug_dim, cell_dim, dose_enc_id, arch, hidden=hidden, dropout=dropout)
    total = sum(p.numel() for p in m.parameters())
    enc = sum(p.numel() for p in m.dose_enc.parameters())
    return total, enc


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed 配置目录（含 pairs.parquet）")
    ap.add_argument("--dose-enc", required=True, choices=list(DOSE_ENCS))
    ap.add_argument("--arch", required=True, choices=["A1", "A2", "A3", "A4"])
    ap.add_argument("--protocol", required=True, choices=list(PROTOCOLS),
                    help="实体留出或INTERP/EXTRAP/EXTRAP2剂量留出")
    ap.add_argument("--kfold", type=int, default=5, help="LCO/LODO/LDO 的折数")
    ap.add_argument("--fold", type=int, default=0, help="LCO/LODO/LDO 第几折作测试集")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="剂量协议：留出目标点的验证药对比例；实体协议：训练池的验证比例")
    ap.add_argument("--save-pred", action="store_true",
                    help="保存测试集预测（供机制分析）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="独立的数据划分种子；默认沿用模型初始化种子")
    ap.add_argument("--shared-backbone-init", action="store_true",
                    help="剂量层使用独立随机状态，使相同宽度的共同预测部分初始化一致")
    ap.add_argument("--dose-input-scale", choices=["none", "train_standardized"], default="none",
                    help="保留对数剂量，或按训练剂量的合并均值和标准差变换")
    ap.add_argument("--epochs", type=int, default=1000,
                    help="**上限**轮数（早停会提前结束）")
    ap.add_argument("--patience", type=int, default=50,
                    help="早停：验证 RMSE 连续多少轮无改善即停止")
    ap.add_argument("--min-delta", type=float, default=0.0, help="判定「改善」的最小幅度")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    # ---- 周期编码的频率定标（见各编码器的 docstring）----
    ap.add_argument("--sigma", type=float, default=0.25,
                    help="D3/D4 的 Fourier 频率标准差（Tancik 定标：σ≤1/(4Δx)=0.25）")
    ap.add_argument("--w0", type=float, default=7.0,
                    help="D5 SIREN 的频率（SIREN 定标：ω₀=30/(输入跨度/2)≈7）")
    ap.add_argument("--subset", type=float, default=1.0, help="采样比例（调试用）")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drug-feat", default="morgan",
                    choices=["onehot", "rdkit", "morgan", "chemberta"])
    ap.add_argument("--cell-feat", default="expr", choices=["expr", "onehot"],
                    help="细胞系表征：expr=表达谱（默认）/ onehot=独热（无生物学信息，对照用）")
    ap.add_argument("--shuffle-drug-cell", action="store_true",
                    help="随机打乱药物↔细胞系配对（防 shortcut 对照）")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    cfg = os.path.basename(os.path.normpath(args.data))

    suffix = ""
    if args.drug_feat != "morgan":
        suffix += f"-{args.drug_feat}"
    if args.cell_feat != "expr":
        suffix += f"-cell{args.cell_feat}"
    if args.shuffle_drug_cell:
        suffix += f"-shuf{args.shuffle_seed}"
    if split_seed != args.seed:
        suffix += f"-split{split_seed}"
    if args.shared_backbone_init:
        suffix += "-sharedinit"
    if args.dose_input_scale != "none":
        suffix += f"-dosescale{args.dose_input_scale}"
    fold_tag = f"-f{args.fold}" if args.protocol in FOLD_PROTOCOLS else ""
    tag = f"{cfg}-{args.dose_enc}-{args.arch}-{args.protocol}-s{args.seed}{fold_tag}{suffix}"
    run_dir = os.path.join(args.out, f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{tag}")
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "train.log"), "w", encoding="utf-8")

    def log(m):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"=== {tag} ===")
    log(f"args: {vars(args)}")

    D = args.data
    df = pd.read_parquet(os.path.join(D, "pairs.parquet"))
    if args.subset < 1.0:
        df = df.sample(frac=args.subset, random_state=args.seed).reset_index(drop=True)
    log(f"数据 {len(df):,} 行  配置 {cfg}")

    n_drug = int(max(df["DRUG1_IDX"].max(), df["DRUG2_IDX"].max())) + 1
    repr_path = os.path.join(D, "drug_reprs.npz")
    if os.path.exists(repr_path):
        _z = np.load(repr_path, allow_pickle=True)
        drf = _z[args.drug_feat].astype(np.float32)
        log(f"药物表征 = {args.drug_feat}  {drf.shape}  (来自 drug_reprs.npz)")
    elif args.drug_feat == "onehot":
        drf = np.eye(n_drug, dtype=np.float32)
        log(f"药物表征 = one-hot  {drf.shape}  (现场构造)")
    else:
        drf = np.load(os.path.join(D, "drug_features.npz"),
                      allow_pickle=True)["morgan"].astype(np.float32)
        log(f"药物表征 = morgan  {drf.shape}  (回退到 drug_features.npz)")
    assert drf.shape[0] == n_drug, f"药物数与表征行数不符: {drf.shape[0]} vs {n_drug}"

    cel = np.load(os.path.join(D, "cell_expression.npy")).astype(np.float32)
    if args.cell_feat == "onehot":
        # 对照：把细胞系换成独热向量，剥离全部生物学信息
        cel = np.eye(cel.shape[0], dtype=np.float32)
        log(f"细胞系表征 = one-hot  {cel.shape}（无生物学信息）")
    else:
        cel = (cel - cel.mean(0, keepdims=True)) / (cel.std(0, keepdims=True) + 1e-6)
    log(f"药物特征 {drf.shape}  细胞系特征 {cel.shape}")

    if args.shuffle_drug_cell:
        rs = np.random.RandomState(args.shuffle_seed)
        perm = rs.permutation(len(df))
        for c in ["DRUG1_IDX", "DRUG2_IDX"]:
            df[c] = df[c].values[perm]
        log(f"药物↔细胞系配对已打乱（shuffle_seed={args.shuffle_seed}）")

    tr, va, te = make_split(df, args.protocol, split_seed, args.kfold, args.fold,
                            args.val_frac)
    if len(te) == 0:
        raise RuntimeError(f"{args.protocol} fold={args.fold} 测试集为空")
    log(f"划分 train/val/test = {len(tr):,}/{len(va):,}/{len(te):,} "
        f"({len(tr)/len(df)*100:.1f}/{len(va)/len(df)*100:.1f}/{len(te)/len(df)*100:.1f} %)")
    np.savez_compressed(os.path.join(run_dir, "split_indices.npz"),
                        train=tr, val=va, test=te)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"设备 {dev}")

    df = df.reset_index(drop=True)
    np_idx = {c: df[c].values.astype(np.int64)
              for c in ["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX"]}
    np_x1 = np.nan_to_num(df["CONC1_LOG"].values.astype(np.float32), nan=-9.0)
    np_x2 = np.nan_to_num(df["CONC2_LOG"].values.astype(np.float32), nan=-9.0)
    np_y = df["PERCENTGROWTH"].values.astype(np.float32)

    drug_t = torch.from_numpy(np.ascontiguousarray(drf)).to(dev)
    cell_t = torch.from_numpy(np.ascontiguousarray(cel)).to(dev)
    log(f"嵌入表常驻显存: 药物 {drug_t.numel()*4/1e6:.2f} MB, "
        f"细胞系 {cell_t.numel()*4/1e6:.2f} MB")

    class Split:
        def __init__(self, ix):
            ix = np.asarray(ix)
            self.n = len(ix)
            self.i1 = torch.from_numpy(np.ascontiguousarray(np_idx["DRUG1_IDX"][ix])).to(dev)
            self.i2 = torch.from_numpy(np.ascontiguousarray(np_idx["DRUG2_IDX"][ix])).to(dev)
            self.ic = torch.from_numpy(np.ascontiguousarray(np_idx["CELL_IDX"][ix])).to(dev)
            self.x1 = torch.from_numpy(np.ascontiguousarray(np_x1[ix])).to(dev)
            self.x2 = torch.from_numpy(np.ascontiguousarray(np_x2[ix])).to(dev)
            self.y = torch.from_numpy(np.ascontiguousarray(np_y[ix])).to(dev)

        def batches(self, bs, shuffle=False):
            order = (torch.randperm(self.n, device=dev)
                     if shuffle else torch.arange(self.n, device=dev))
            for s in range(0, self.n, bs):
                sl = order[s:s + bs]
                yield (drug_t[self.i1[sl]], drug_t[self.i2[sl]], cell_t[self.ic[sl]],
                       self.x1[sl], self.x2[sl], self.y[sl])

    TR, VA, TE = Split(tr), Split(va), Split(te)

    dose_transform = fit_dose_input_transform(np_x1, np_x2, tr, args.dose_input_scale)
    log(f"剂量输入变换（只拟合训练剂量）: {dose_transform}")
    model = ComboModel(drf.shape[1], cel.shape[1], args.dose_enc, args.arch,
                       hidden=args.hidden, dropout=args.dropout,
                       sigma=args.sigma, w0=args.w0,
                       shared_backbone_init=args.shared_backbone_init,
                       dose_input_scale=args.dose_input_scale,
                       dose_input_mean=dose_transform["mean"], dose_input_std=dose_transform["std"]).to(dev)
    affine = getattr(model.dose_enc, "lin", getattr(model.dose_enc, "fc", None))
    initial_hashes = dict(drug_projection=module_hash(model.drug_proj),
                          cell_projection=module_hash(model.cell_proj),
                          prediction_network=module_hash(model.head),
                          dose_affine=module_hash(affine),
                          dose_network=module_hash(getattr(model.dose_enc, "mlp", None)))
    if args.arch == "A4":
        initial_hashes["dose_drug_fusion"] = module_hash(model.dose_drug_fusion)
    n_par = sum(p.numel() for p in model.parameters())
    n_par_enc = sum(p.numel() for p in model.dose_enc.parameters())
    log(f"模型参数量 {n_par:,}（其中剂量编码 {n_par_enc:,}）")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()

    # ---- 早停 ----
    # 监控验证集 RMSE，连续 patience 轮无改善即停；
    # 始终恢复验证集最优的权重（best.pt）。
    # 预算参照 MatchMaker（最多 1000 轮、patience 100）；此处用 patience 50。
    best = {"val_rmse": float("inf"), "epoch": -1}
    best_ep = 0
    stopped_ep = None
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for d1, d2, c, x1, x2, y in TR.batches(args.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            pred = model(d1, d2, c, x1, x2)
            loss = lossf(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for d1, d2, c, x1, x2, y in VA.batches(args.batch_size * 4):
                vp.append(model(d1, d2, c, x1, x2).cpu().numpy()); vy.append(y.cpu().numpy())
        vm = metrics(np.concatenate(vy), np.concatenate(vp))
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1),
                     **{f"val_{k}": v for k, v in vm.items()}})
        log(f"ep{ep:3d}  train_loss={tot/max(nb,1):8.3f}  "
            f"val_rmse={vm['rmse']:7.3f}  val_r2={vm['r2']:6.3f}")

        if vm["rmse"] < best["val_rmse"] - args.min_delta:
            best = {"val_rmse": vm["rmse"], "epoch": ep}
            best_ep = ep
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

        if ep - best_ep >= args.patience:
            stopped_ep = ep
            log(f"早停 @ep{ep}：已 {args.patience} 轮无改善（最优在 ep{best_ep}）")
            break

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=dev))
    model.eval(); tp, ty = [], []
    with torch.no_grad():
        for d1, d2, c, x1, x2, y in TE.batches(args.batch_size * 4):
            tp.append(model(d1, d2, c, x1, x2).cpu().numpy()); ty.append(y.cpu().numpy())
    tp, ty = np.concatenate(tp), np.concatenate(ty)
    tm = metrics(ty, tp)

    if args.save_pred:
        te_df = df.iloc[te]
        # 实体键可能是 int（ALMANAC 的 NSC）也可能是 str（O'Neil 的药物名）→ 统一转字符串
        np.savez_compressed(
            os.path.join(run_dir, "test_pred.npz"),
            y_true=ty.astype(np.float32),
            y_pred=tp.astype(np.float32),
            row_index=te.astype(np.int64),
            drug1=te_df["DRUG1_IDX"].values.astype(np.int32),
            drug2=te_df["DRUG2_IDX"].values.astype(np.int32),
            cell=te_df["CELL_IDX"].values.astype(np.int32),
            drug1_key=te_df["NSC1"].to_numpy(dtype=str),
            drug2_key=te_df["NSC2"].to_numpy(dtype=str),
            conc1_log=te_df["CONC1_LOG"].values.astype(np.float32),
            conc2_log=te_df["CONC2_LOG"].values.astype(np.float32),
            conc1_pos=te_df["CONC1_LOG_POS"].values.astype(np.float32),
            conc2_pos=te_df["CONC2_LOG_POS"].values.astype(np.float32),
            expected=te_df["EXPECTEDGROWTH"].values.astype(np.float32),
            cellname=te_df["CELLNAME"].to_numpy(dtype=str),
            panel=te_df["PANEL"].to_numpy(dtype=str),
        )
        log(f"测试集预测已保存: test_pred.npz ({len(ty):,} 条)")

    res = {"tag": tag, "config": cfg, "args": vars(args),
           "split_seed": split_seed,
           "dose_input_transform": dose_transform, "initialization_hashes": initial_hashes,
           "split_version": DOSE_SPLIT_VERSION if args.protocol in GRID_PROTOCOLS else "entity-v2",
           "n_params": n_par, "n_params_dose_enc": n_par_enc,
           "best_epoch": best["epoch"],
           "n_epochs_run": len(hist), "stopped_epoch": stopped_ep,
           "early_stopped": stopped_ep is not None,
           "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "test": tm, "best_val_rmse": best["val_rmse"]}
    json.dump(res, open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(hist, open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8"),
              indent=1)

    log("")
    log(f"=== 测试集结果（best epoch {best['epoch']}） ===")
    for k, v in tm.items():
        log(f"   {k:10s} {v:.4f}" if isinstance(v, float) else f"   {k:10s} {v}")
    log(f"结果目录: {run_dir}")
    logf.close()


if __name__ == "__main__":
    main()
