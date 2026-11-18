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

from src.models.dose_encoders import DOSE_ENCS, fit_dose_input_transform, module_hash
from src.models.fusion_networks import ComboModel


def grid_structure(df):
    """Evaluate surface completeness from distinct dose pairs; replicates do not alter grid size."""
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
    """Mark interior centre dose points for each complete (drug pair, cell line) grid.

    Centre ranks are defined as lo = (n + 1) // 2, hi = (n + 2) // 2:
      n=3 -> {2}    (1 point per surface; NCI-ALMANAC 3x3)
      n=4 -> {2, 3} (4 points per surface; O'Neil 4x4)
      n=5 -> {3}    (1 point per surface)

    Applies only to complete grids with min(n1, n2) >= 3.
    """
    r1, r2, n1, n2, complete = grid_structure(df)
    lo1, hi1 = (n1 + 1) // 2, (n1 + 2) // 2
    lo2, hi2 = (n2 + 1) // 2, (n2 + 2) // 2
    in1 = (r1 >= lo1) & (r1 <= hi1)
    in2 = (r2 >= lo2) & (r2 <= hi2)
    return (complete & in1 & in2).fillna(False).values


#: Protocols with fold-based entity holdouts
FOLD_PROTOCOLS = ("LCO", "LODO", "LDO")
#: Protocols requiring complete dose grids
GRID_PROTOCOLS = ("INTERP", "EXTRAP", "EXTRAP2")
DOSE_SPLIT_VERSION = "known-curve-dose-v3"
#: Supported split protocols
PROTOCOLS = ("LTO", "LPO", "LCO", "LODO", "LDO", "INTERP", "EXTRAP", "EXTRAP2")


def _entity_keys(df):
    """Return drug entity identifiers (integer NSC codes for ALMANAC, names for O'Neil)."""
    return df["NSC1"].values, df["NSC2"].values


def _pool_split(pair_codes, m_pool, m_te, rng, val_frac=0.25):
    """Partition the non-test pool into train and validation sets grouped by drug pair."""
    pool_codes = pair_codes[m_pool]
    uniq = np.unique(pool_codes)
    rng.shuffle(uniq)
    n_va = max(1, int(val_frac * len(uniq)))
    va_u = set(uniq[:n_va].tolist())
    m_va = m_pool & np.isin(pair_codes, list(va_u))
    m_tr = m_pool & ~m_va
    return m_tr, m_va, m_te


def make_split(df, protocol, seed=42, kfold=5, fold=0, val_frac=0.25):
    """Return disjoint (train_idx, val_idx, test_idx) arrays for the given protocol.

    Grid holdouts assign held-out points at the surface level and partition
    held-out pairs between validation and test sets while keeping outer
    training points across all pairs in the training pool.
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
        uniq = np.unique(unit)
        rng.shuffle(uniq)
        n_tr = int(0.6 * len(uniq))
        n_va = int(0.2 * len(uniq))
        m_tr = np.isin(unit, uniq[:n_tr])
        m_va = np.isin(unit, uniq[n_tr:n_tr + n_va])
        return idx[m_tr], idx[m_va], idx[~(m_tr | m_va)]

    if protocol == "LPO":
        uniq = np.unique(pair_codes)
        rng.shuffle(uniq)
        n_tr = int(0.6 * len(uniq))
        n_va = int(0.2 * len(uniq))
        m_tr = np.isin(pair_codes, uniq[:n_tr])
        m_va = np.isin(pair_codes, uniq[n_tr:n_tr + n_va])
        return idx[m_tr], idx[m_va], idx[~(m_tr | m_va)]

    if protocol in GRID_PROTOCOLS:
        if not 0 < val_frac < 1:
            raise ValueError("Validation pair fraction for dose holdouts must lie strictly between 0 and 1")
        if protocol == "INTERP":
            held = grid_center_mask(df)
        else:
            r1, r2, n1, n2, complete = grid_structure(df)
            at_high = (r1 == n1) if protocol == "EXTRAP" else (r2 == n2)
            held = (complete & at_high).fillna(False).to_numpy()
        eligible_pairs = np.unique(pair_codes[held])
        if len(eligible_pairs) < 2:
            raise ValueError("Dose holdout requires at least two pairs with complete surfaces to separate validation and test")
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
            in_h1 = np.isin(d1, held)
            in_h2 = np.isin(d2, held)
            if protocol == "LODO":
                m_te = in_h1 | in_h2
                m_pool = ~m_te
            else:  # LDO
                m_te = in_h1 & in_h2
                train_drugs = np.setdiff1d(ents, held)
                m_pool = (np.isin(d1, train_drugs) & np.isin(d2, train_drugs))
        m_tr, m_va, _ = _pool_split(pair_codes, m_pool, m_te, rng, val_frac)
        return idx[m_tr], idx[m_va], idx[m_te]

    raise NotImplementedError(f"Protocol {protocol} is not implemented")


def metrics(y, p):
    """Compute RMSE, MAE, Pearson, Spearman, R^2, and sample count."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = y - p
    out = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}
    if y.std() > 0 and p.std() > 0:
        out["pearson"] = float(np.corrcoef(y, p)[0, 1])
        out["spearman"] = float(np.corrcoef(np.argsort(np.argsort(y)),
                                            np.argsort(np.argsort(p)))[0, 1])
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    out["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    out["n"] = int(len(y))
    return out


def count_params(dose_enc_id, arch="A1", drug_dim=2048, cell_dim=977,
                 hidden=512, dropout=0.2):
    """Return (total_params, dose_encoder_params) for a given encoder and architecture."""
    m = ComboModel(drug_dim, cell_dim, dose_enc_id, arch, hidden=hidden, dropout=dropout)
    total = sum(p.numel() for p in m.parameters())
    enc = sum(p.numel() for p in m.dose_enc.parameters())
    return total, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Processed dataset directory containing pairs.parquet")
    ap.add_argument("--dose-enc", required=True, choices=list(DOSE_ENCS))
    ap.add_argument("--arch", required=True, choices=["A1", "A2", "A3", "A4"])
    ap.add_argument("--protocol", required=True, choices=list(PROTOCOLS),
                    help="Entity or dose-grid holdout protocol (INTERP/EXTRAP/EXTRAP2)")
    ap.add_argument("--kfold", type=int, default=5, help="Number of folds for LCO/LODO/LDO")
    ap.add_argument("--fold", type=int, default=0, help="Test fold index for LCO/LODO/LDO")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="Fraction of eligible pairs or pool assigned to validation")
    ap.add_argument("--save-pred", action="store_true",
                    help="Save test-set predictions to test_pred.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="Optional split seed; defaults to model initialization seed")
    ap.add_argument("--shared-backbone-init", action="store_true",
                    help="Initialize the dose branch under an isolated RNG state for matched backbones")
    ap.add_argument("--dose-input-scale", choices=["none", "train_standardized"], default="none",
                    help="Keep raw log10(uM) doses or standardize using training-split statistics")
    ap.add_argument("--epochs", type=int, default=1000,
                    help="Maximum training epochs")
    ap.add_argument("--patience", type=int, default=50,
                    help="Early-stopping patience on validation RMSE")
    ap.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation RMSE improvement")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.25,
                    help="Random Fourier frequency scale for D3/D4 (default 0.25)")
    ap.add_argument("--w0", type=float, default=7.0,
                    help="SIREN frequency scale w0 for D5 (default 7.0)")
    ap.add_argument("--subset", type=float, default=1.0, help="Optional subsampling fraction for debugging")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drug-feat", default="morgan",
                    choices=["onehot", "rdkit", "morgan", "chemberta"])
    ap.add_argument("--cell-feat", default="expr", choices=["expr", "onehot"],
                    help="Cell-line representation: expr (default) or onehot")
    ap.add_argument("--shuffle-drug-cell", action="store_true",
                    help="Randomly permute drug-cell pairings as a shortcut control")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"=== {tag} ===")
    log(f"args: {vars(args)}")

    D = args.data
    df = pd.read_parquet(os.path.join(D, "pairs.parquet"))
    if args.subset < 1.0:
        df = df.sample(frac=args.subset, random_state=args.seed).reset_index(drop=True)
    log(f"rows={len(df):,} dataset={cfg}")

    n_drug = int(max(df["DRUG1_IDX"].max(), df["DRUG2_IDX"].max())) + 1
    repr_path = os.path.join(D, "drug_reprs.npz")
    if os.path.exists(repr_path):
        _z = np.load(repr_path, allow_pickle=True)
        drf = _z[args.drug_feat].astype(np.float32)
        log(f"drug_feat={args.drug_feat} shape={drf.shape} (from drug_reprs.npz)")
    elif args.drug_feat == "onehot":
        drf = np.eye(n_drug, dtype=np.float32)
        log(f"drug_feat=onehot shape={drf.shape} (constructed on the fly)")
    else:
        drf = np.load(os.path.join(D, "drug_features.npz"),
                      allow_pickle=True)["morgan"].astype(np.float32)
        log(f"drug_feat=morgan shape={drf.shape} (fallback drug_features.npz)")
    assert drf.shape[0] == n_drug, f"Drug count mismatch: {drf.shape[0]} vs {n_drug}"

    cel = np.load(os.path.join(D, "cell_expression.npy")).astype(np.float32)
    if args.cell_feat == "onehot":
        cel = np.eye(cel.shape[0], dtype=np.float32)
        log(f"cell_feat=onehot shape={cel.shape}")
    else:
        cel = (cel - cel.mean(0, keepdims=True)) / (cel.std(0, keepdims=True) + 1e-6)
    log(f"drug_features={drf.shape} cell_features={cel.shape}")

    if args.shuffle_drug_cell:
        rs = np.random.RandomState(args.shuffle_seed)
        perm = rs.permutation(len(df))
        for c in ["DRUG1_IDX", "DRUG2_IDX"]:
            df[c] = df[c].values[perm]
        log(f"Shuffled drug-cell pairings (shuffle_seed={args.shuffle_seed})")

    tr, va, te = make_split(df, args.protocol, split_seed, args.kfold, args.fold,
                            args.val_frac)
    if len(te) == 0:
        raise RuntimeError(f"Empty test split for {args.protocol} fold={args.fold}")
    log(f"split train/val/test = {len(tr):,}/{len(va):,}/{len(te):,} "
        f"({len(tr)/len(df)*100:.1f}/{len(va)/len(df)*100:.1f}/{len(te)/len(df)*100:.1f} %)")
    np.savez_compressed(os.path.join(run_dir, "split_indices.npz"),
                        train=tr, val=va, test=te)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"device={dev}")

    df = df.reset_index(drop=True)
    np_idx = {c: df[c].values.astype(np.int64)
              for c in ["DRUG1_IDX", "DRUG2_IDX", "CELL_IDX"]}
    np_x1 = np.nan_to_num(df["CONC1_LOG"].values.astype(np.float32), nan=-9.0)
    np_x2 = np.nan_to_num(df["CONC2_LOG"].values.astype(np.float32), nan=-9.0)
    np_y = df["PERCENTGROWTH"].values.astype(np.float32)

    drug_t = torch.from_numpy(np.ascontiguousarray(drf)).to(dev)
    cell_t = torch.from_numpy(np.ascontiguousarray(cel)).to(dev)
    log(f"embedding_tables_mb: drug={drug_t.numel()*4/1e6:.2f} cell={cell_t.numel()*4/1e6:.2f}")

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
    log(f"dose_input_transform (fit on training split only): {dose_transform}")
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
    log(f"params total={n_par:,} dose_encoder={n_par_enc:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()

    best = {"val_rmse": float("inf"), "epoch": -1}
    best_ep = 0
    stopped_ep = None
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        nb = 0
        for d1, d2, c, x1, x2, y in TR.batches(args.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            pred = model(d1, d2, c, x1, x2)
            loss = lossf(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for d1, d2, c, x1, x2, y in VA.batches(args.batch_size * 4):
                vp.append(model(d1, d2, c, x1, x2).cpu().numpy())
                vy.append(y.cpu().numpy())
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
            log(f"Early stopping @ep{ep}: no improvement for {args.patience} epochs (best at ep{best_ep})")
            break

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=dev))
    model.eval()
    tp, ty = [], []
    with torch.no_grad():
        for d1, d2, c, x1, x2, y in TE.batches(args.batch_size * 4):
            tp.append(model(d1, d2, c, x1, x2).cpu().numpy())
            ty.append(y.cpu().numpy())
    tp, ty = np.concatenate(tp), np.concatenate(ty)
    tm = metrics(ty, tp)

    if args.save_pred:
        te_df = df.iloc[te]
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
        log(f"Saved test predictions: test_pred.npz ({len(ty):,} rows)")

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
    log(f"=== Test metrics (best epoch {best['epoch']}) ===")
    for k, v in tm.items():
        log(f"   {k:10s} {v:.4f}" if isinstance(v, float) else f"   {k:10s} {v}")
    log(f"Output directory: {run_dir}")
    logf.close()


if __name__ == "__main__":
    main()
