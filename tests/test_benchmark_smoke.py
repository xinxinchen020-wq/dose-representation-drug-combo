"""Synthetic end-to-end smoke tests requiring no external datasets or GPU."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]


class BenchmarkSmokeTests(unittest.TestCase):
    def test_preprocess_default_paths_point_to_repo_root(self):
        from src.preprocess import convert_oneil_workbook, prepare_almanac_reference, prepare_oneil_reference

        self.assertEqual(Path(convert_oneil_workbook.ROOT), REPO)
        self.assertEqual(Path(convert_oneil_workbook.DEFAULT_SRC), REPO / "data" / "raw" / "oneil" / "oneil_sup.xlsx")
        self.assertEqual(Path(prepare_oneil_reference.ROOT), REPO)
        self.assertEqual(Path(prepare_almanac_reference.ROOT), REPO)
        self.assertTrue((REPO / "configs" / "gene_panel_l1000.json").is_file())

    def test_dose_encoder_shapes_and_parameter_counts(self):
        from src.models.fusion_networks import ComboModel

        expected = {
            ("raw64_fixed", "A1"): (0, 1_175_041),
            ("raw64_fixed", "A4"): (0, 1_019_521),
            ("D1", "A1"): (0, 1_175_041),
            ("linear64", "A1"): (128, 1_175_169),
            ("sin_affine64", "A1"): (128, 1_175_169),
            ("linear64_mlp", "A1"): (16_704, 1_191_745),
            ("linear64_mlp", "A4"): (16_704, 1_036_225),
            ("sin_affine64_mlp", "A1"): (16_704, 1_191_745),
            ("D1u", "A1"): (16_576, 1_191_617),
            ("D3", "A1"): (16_576, 1_191_617),
            ("D3", "A4"): (16_576, 1_036_097),
            ("compact_scalar_mlp", "A1"): (8_512, 1_183_553),
        }
        d1 = torch.zeros(4, 2048)
        d2 = torch.zeros(4, 2048)
        cell = torch.zeros(4, 977)
        x1 = torch.tensor([-2.0, -1.0, 0.0, 1.0])
        x2 = torch.tensor([-1.5, -0.5, 0.5, 1.5])
        for (method, arch), (exp_enc, exp_tot) in expected.items():
            model = ComboModel(2048, 977, method, arch)
            self.assertEqual(sum(p.numel() for p in model.dose_enc.parameters()), exp_enc)
            self.assertEqual(sum(p.numel() for p in model.parameters()), exp_tot)
            out = model(d1, d2, cell, x1, x2)
            self.assertEqual(tuple(out.shape), (4,))

    def test_synthetic_splits_training_and_lightgbm_pipeline(self):
        from src.eval.lightgbm_comparison import paired_intervals
        from src.splits.make_holdout_splits import build_splits
        from src.train.lightgbm_comparison import ENCODINGS, dose_features

        cfg = yaml.safe_load((REPO / "configs" / "default_benchmark.yaml").read_text(encoding="utf-8"))
        doses = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
        rows = []
        pairs = [("D0", "D1"), ("D0", "D2"), ("D1", "D2"), ("D2", "D3")]
        pair_map = {"D0": 0, "D1": 1, "D2": 2, "D3": 3}
        for c_idx, cell in enumerate(["CELL_A", "CELL_B"]):
            for dA, dB in pairs:
                for i, c1 in enumerate(doses):
                    for j, c2 in enumerate(doses):
                        rows.append({
                            "DRUG1_IDX": pair_map[dA],
                            "DRUG2_IDX": pair_map[dB],
                            "CELL_IDX": c_idx,
                            "CONC1": float(10.0 ** c1),
                            "CONC2": float(10.0 ** c2),
                            "CONC1_LOG": float(c1),
                            "CONC2_LOG": float(c2),
                            "CONC1_LOG_POS": float(i / 3.0),
                            "CONC2_LOG_POS": float(j / 3.0),
                            "PERCENTGROWTH": float(50.0 - 8.0 * c1 - 6.0 * c2 + 0.5 * c_idx),
                            "EXPECTEDGROWTH": float(50.0 - 8.0 * c1 - 6.0 * c2),
                            "SCORE": 0.0,
                            "SCREENER": "SYNTH",
                            "PANEL": "TEST",
                            "PAIR": f"{dA}_{dB}",
                            "NSC1": dA,
                            "NSC2": dB,
                            "CELLNAME": cell,
                        })
        frame = pd.DataFrame(rows)
        splits = dict(build_splits(frame, "synthetic", cfg["tasks"], pair_seed=0, random_seed=20261016, validation_fraction=0.25))
        self.assertEqual(set(splits), {"centre", "random", "drug1_high_dose", "drug2_high_dose"})
        for task, (tr, va, te) in splits.items():
            combined = np.concatenate([tr, va, te])
            self.assertEqual(len(combined), len(frame))
            self.assertTrue(np.array_equal(np.sort(combined), np.arange(len(frame))), msg=f"Invalid partition in {task}")
            self.assertFalse(set(frame.iloc[va].PAIR) & set(frame.iloc[te].PAIR), msg=f"Pair overlap in {task}")

        # Verify no grid corners held out in within-range random holdout
        va_te = np.concatenate(splits["random"][1:])
        c1_vals = frame.iloc[va_te].CONC1_LOG.to_numpy()
        c2_vals = frame.iloc[va_te].CONC2_LOG.to_numpy()
        at_corner = ((c1_vals == -2.0) | (c1_vals == 1.0)) & ((c2_vals == -2.0) | (c2_vals == 1.0))
        self.assertFalse(at_corner.any())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "synth"
            data_dir.mkdir()
            frame.to_parquet(data_dir / "pairs.parquet", index=False)
            rng = np.random.default_rng(0)
            np.savez(data_dir / "drug_reprs.npz", morgan=rng.integers(0, 2, size=(4, 16)).astype(np.float32))
            np.save(data_dir / "cell_expression.npy", rng.normal(size=(2, 12)).astype(np.float32))

            idx_path = root / "split_indices.npz"
            tr, va, te = splits["centre"]
            np.savez(idx_path, train=tr, val=va, test=te)

            out_dir = root / "fit_out"
            cmd = [
                sys.executable, "-m", "src.train.run_fit",
                "--indices", str(idx_path),
                "--data", str(data_dir),
                "--out", str(out_dir),
                "--dose-enc", "raw64_fixed",
                "--arch", "A1",
                "--protocol", "INTERP",
                "--seed", "0",
                "--epochs", "2",
                "--batch-size", "32",
                "--hidden", "32",
                "--device", "cpu",
                "--save-pred",
            ]
            subprocess.run(cmd, cwd=str(REPO), check=True, stdout=subprocess.DEVNULL)
            run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            for artifact in ("results.json", "history.json", "best.pt", "test_pred.npz"):
                self.assertTrue((run_dir / artifact).is_file(), msg=f"Missing {artifact}")
            res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(res["n_test"], len(te))
            with np.load(run_dir / "test_pred.npz", allow_pickle=False) as pred:
                self.assertTrue(np.array_equal(pred["row_index"], te))

        # Smoke check LightGBM encoding features + training + bootstrap interval computation
        for enc in ENCODINGS:
            feats = dose_features(frame, enc)
            self.assertEqual(feats.shape[0], len(frame))
            self.assertEqual(feats.shape[1], 2 if enc == "direct_log_dose" else 128)
        ds = lgb.Dataset(np.column_stack([frame.CONC1_LOG, frame.CONC2_LOG]), label=frame.PERCENTGROWTH.to_numpy())
        booster = lgb.train({"objective": "regression", "verbosity": -1, "num_leaves": 7, "min_data_in_leaf": 4}, ds, num_boost_round=5)
        self.assertEqual(len(booster.predict(np.zeros((3, 2)))), 3)

        err_df = pd.DataFrame({
            "PAIR": frame.iloc[te].PAIR.values,
            "direct_log_dose": rng.uniform(0.1, 2.0, size=len(te)),
            "fixed_sinusoidal": rng.uniform(0.1, 2.0, size=len(te)),
            "fixed_random_fourier": rng.uniform(0.1, 2.0, size=len(te)),
        })
        intervals = paired_intervals("centre", err_df)
        self.assertEqual(len(intervals), 2)


if __name__ == "__main__":
    unittest.main()
