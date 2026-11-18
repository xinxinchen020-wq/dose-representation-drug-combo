# Benchmarking dose representations in drug-combination response prediction exposes extrapolation degradation and aggregation artefacts

Research code for the controlled comparison of eight dose representations in the O’Neil and NCI-ALMANAC drug-combination screens. The benchmark evaluates held-out doses for known drug pairs and cell lines using centre, within-range random, first-drug high-dose and second-drug high-dose holdouts. Extended analyses evaluate within-drug fusion, a compact scalar dose network, an internal follow-up split, cross-family LightGBM controls, surface-level error decompositions and ComboScore ranking utility.

## Environment

```bash
conda env create -f environment.yml
conda activate dose-rep-bench
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Alternatively, install the Python dependencies via `pip install -r requirements.txt` together with the [appropriate PyTorch wheel](https://pytorch.org/get-started/previous-versions/) for your platform. The recorded runtime snapshot lists Python 3.12.15, PyTorch 2.6.0+cu124, NumPy 2.5.3, pandas 3.0.6 and LightGBM 4.7.0. The training runner accepts `--device cpu` for small checks. (`transformers` is only needed if invoking optional `--chemberta` embedding extraction in `src.preprocess.make_drug_reprs`, which is not used in the manuscript benchmark.)

## Quick verification

Run the synthetic end-to-end smoke suite without downloading raw datasets:

```bash
python -m unittest discover -s tests -v
```

## Obtain the input files

- **O’Neil screen:** the combination response workbook in the [O’Neil et al. supplementary materials](https://doi.org/10.1158/1535-7163.MCT-15-0843). Save the workbook as `data/raw/oneil/oneil_sup.xlsx`.
- **NCI-ALMANAC:** the November 2017 combination growth data from the [NCI download page](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC). Save the uncompressed file as `data/raw/almanac/NCL_Combo_Response.csv`. Supply the matching NSC-to-SMILES table as `data/raw/almanac/nsc_smiles.csv`, with columns `NSC` and `SMILES`.
- **Cell-line expression:** `data/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` from the [DepMap Portal](https://depmap.org/portal/data_page/). The preprocessing records identify the input as release 25Q3 (SHA-256 `9d7e64ebbcb2811fa5e0ecf56d952ee96ca9f1f2b6ead00bb17480c1c01848d1`).
- **Gene annotation:** supply `data/raw/gene_lists/hgnc_complete_set.txt` from [HGNC](https://www.genenames.org/download/archive/). The repository includes the ordered panel of 977 genes derived from the landmark annotations in [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742).
- **O’Neil identifiers:** supply `DrugComb_cell_line_identifiers.xlsx` and `DrugComb_drug_identifiers.xlsx` under `data/raw/drugcomb/`. Reference preparation uses the cell-line and drug identifier sheets from [DrugComb](https://drugcomb.org/). The repository includes the experiment's cell-line mappings and additional oxaliplatin structure.

Use the same input snapshots when comparing numerical results with the manuscript. The source datasets are subject to their providers' terms. The MIT licence applies to the code in this repository.

## Prepare the two datasets

Run the following from the repository root. Reference preparation builds the entity dictionaries and aligned expression arrays. The final two commands select the complete experimental grids used for analysis.

```bash
python -m src.preprocess.convert_oneil_workbook
python -m src.preprocess.prepare_oneil_reference --out data/reference/oneil --cell-subset configs/cells_oneil.json --configs-dir configs --gene-panel configs/gene_panel_l1000.json --extra-smiles configs/extra_smiles.json
python -m src.preprocess.prepare_almanac_reference --out data/reference/almanac_dm47 --cell-source depmap --cell-subset configs/cells_almanac_dm47.json --configs-dir configs --gene-panel configs/gene_panel_l1000.json
python -m src.preprocess.extract_features data/reference/oneil data/reference/almanac_dm47
python -m src.preprocess.build_oneil_grids --raw data/raw/oneil/oneil_raw.csv --reference data/reference/oneil --out data/processed/oneil_batch_checked
python -m src.preprocess.build_almanac_grids --raw data/raw/almanac/NCL_Combo_Response.csv --reference data/reference/almanac_dm47 --out data/processed/almanac_dm47_batch_checked
```

If you already have the processed snapshots, place each dataset in its corresponding directory under `data/processed/`. Each directory needs `pairs.parquet`, `drug_reprs.npz` with Morgan fingerprints and `cell_expression.npy`.

## Run the benchmark

```bash
python -m src.splits.make_holdout_splits
python -m src.splits.make_holdout_splits --follow-up
bash scripts/run_all_benchmarks.sh
python -m src.eval.metrics_and_ranks
python -m src.eval.drug_exclusion_decomp
bash scripts/run_extended_ablations.sh
python -m src.eval.metrics_and_ranks --extended --output outputs/extended_metrics.csv
```

To inspect comparison groups without training, run `python -m src.train.train_runner primary --plan` or `python -m src.train.train_runner extended --plan` (or pass `--plan` to `bash scripts/run_all_benchmarks.sh --plan` / `bash scripts/run_extended_ablations.sh --plan`). The primary run comprises 64 groups and 576 candidate fits. The extended run reuses matching primary fits and adds two initialisations at the selected learning rate for the original centre and random conditions. The follow-up comparison uses three initialisations with seeds 1 for pair assignment and 20261018 for dose sampling.

Learning rate is selected separately for each dataset, task, architecture and dose representation using the mean best validation root mean squared error across seeds 0, 1 and 2. Additional seeds 3 and 4 use the selected rate. `configs/default_benchmark.yaml` lists all training settings and maps manuscript terms to identifiers in existing experiment records.

Candidate training logs, learning-rate and LightGBM leaf-selection records, runtime snapshots and follow-up surface-diagnostic and ComboScore summary tables are supplied separately in `Supplementary_Data_1_training_records.zip` alongside the manuscript. Raw data, model weights and individual prediction files are stored locally.

## Supplementary dose-response analyses

Use the original processed snapshots and holdouts prepared above. The surface and synergy analyses read the primary benchmark's validation-selected predictions in `outputs/benchmark`. The LightGBM comparison uses the original O’Neil holdouts and the same Morgan fingerprints and expression features.

```bash
python -m src.eval.dose_response_cases
python -m src.eval.synergy_ranking
python -m src.train.lightgbm_comparison --threads 10
python -m src.eval.lightgbm_comparison --compare-mlp
```

- **Dose–response surfaces and error decomposition (`src.eval.dose_response_cases`):** Selects surfaces nearest the within-group median of the paired squared-error differences and decomposes the NCI-ALMANAC centre error reduction by doxorubicin status, exporting SVG, PDF and 600 dpi PNG figures under `outputs/dose_response_followup/surface_cases/`.
- **Synergy ranking after centre completion (`src.eval.synergy_ranking`):** Replaces one held-out centre response per NCI-ALMANAC surface and ranks drug pairs by the mean ComboScore across cell lines, writing summary tables and per-initialisation metrics.
- **O’Neil LightGBM comparison (`src.train.lightgbm_comparison`, `src.eval.lightgbm_comparison`):** Compares direct log-dose, fixed sinusoidal features and fixed random Fourier features in the centre and second-drug high-dose holdouts (`--compare-mlp` merges original neural benchmarks). Pass `--smoke` to `src.train.lightgbm_comparison` for a quick subset check.

Defaults write generated results and models under `outputs/dose_response_followup/`. Pass `--help` to any module for CLI options.

## Repository layout

- `configs/`: benchmark configuration YAML, cell-line subsets and mappings, L1000 gene panel and supplementary drug SMILES.
- `src/preprocess/`: raw workbook conversion, DepMap/DrugComb alignment, feature extraction and exact grid reconstruction.
- `src/splits/`: deterministic dose-grid holdout split generators (`centre`, `random`, `drug1_high_dose`, `drug2_high_dose`).
- `src/models/`: PyTorch dose encoder modules (`DOSE_ENCS`) and `ComboModel` fusion architectures (`A1`, `A4`).
- `src/train/`: single-fit runner, benchmark orchestrator and LightGBM training pipeline.
- `src/eval/`: metric summaries, leave-one-drug-out decomposition, surface case plots, ComboScore ranking and LightGBM/bootstrap comparison.
- `scripts/`: convenience shell entrypoints for primary and extended benchmark suites.
- `tests/`: synthetic end-to-end smoke tests runnable without external datasets.
