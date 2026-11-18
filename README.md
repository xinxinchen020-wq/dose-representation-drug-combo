# Benchmarking dose representations in drug-combination response prediction exposes extrapolation degradation and aggregation artefacts

Research code for the controlled comparison of eight dose representations in the O’Neil and NCI-ALMANAC drug-combination screens. The benchmark evaluates held-out doses for known drug pairs and cell lines. It uses centre, within-range random, first-drug high-dose and second-drug high-dose holdouts. Extended analyses compare joint concatenation with within-drug fusion and include a compact scalar dose network and a follow-up split.

## Environment

```bash
conda env create -f environment.yml
conda activate dose-rep-bench
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

The recorded runtime snapshot lists Python 3.12.15, PyTorch 2.6.0+cu124, NumPy 2.5.3 and pandas 3.0.6. The Conda file specifies the portable dependencies. Install the [appropriate PyTorch wheel](https://pytorch.org/get-started/previous-versions/) for your platform. The training runner also accepts `--device cpu` for small checks.

## Obtain the input files

- **O’Neil screen:** the combination response workbook in the [O’Neil et al. supplementary materials](https://doi.org/10.1158/1535-7163.MCT-15-0843). Save the workbook as `data/raw/oneil/oneil_sup.xlsx`.
- **NCI-ALMANAC:** the November 2017 combination growth data from the [NCI download page](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC). Save the uncompressed file as `data/raw/almanac/NCL_Combo_Response.csv`. Supply the matching NSC-to-SMILES table as `data/raw/almanac/nsc_smiles.csv`, with columns `NSC` and `SMILES`.
- **Cell-line expression:** `data/raw/depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` from the [DepMap Portal](https://depmap.org/portal/data_page/). The preprocessing records identify the input as release 25Q3. The actual input SHA-256 is `9d7e64ebbcb2811fa5e0ecf56d952ee96ca9f1f2b6ead00bb17480c1c01848d1`.
- **Gene annotation:** supply `data/raw/gene_lists/hgnc_complete_set.txt` from [HGNC](https://www.genenames.org/download/archive/). The repository includes the ordered panel of 977 genes derived from the landmark annotations in [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742).
- **O’Neil identifiers:** supply `DrugComb_cell_line_identifiers.xlsx` and `DrugComb_drug_identifiers.xlsx` under `data/raw/drugcomb/`. The reference preparation uses the cell-line and drug identifier sheets from [DrugComb](https://drugcomb.org/). The repository includes the experiment's cell-line mappings and additional oxaliplatin structure.

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

Use `--plan` with either training script to list comparison groups without training. The primary run comprises 64 groups and 576 candidate fits. The extended run reuses matching primary fits and adds two initialisations at the selected learning rate for the original centre and random conditions. The follow-up comparison uses three initialisations. Its seeds are 1 for pair assignment and 20261018 for dose sampling.

Learning rate is selected separately for each dataset, task, architecture and dose representation using the mean best validation root mean squared error across seeds 0, 1 and 2. Additional seeds 3 and 4 use the selected rate. The default configuration lists the training settings and maps manuscript terms to the identifiers in existing experiment records. Model and split logic come from the frozen research implementation.

Candidate training logs, learning-rate and LightGBM leaf-selection records, runtime snapshots and follow-up surface-diagnostic and ComboScore summary tables are supplied separately in `Supplementary_Data_1_training_records.zip` alongside the manuscript. Raw data, model weights and individual prediction files are stored locally.

## Supplementary dose-response analyses

Use the original processed snapshots and holdouts prepared above. The surface and synergy analyses read the primary benchmark's validation-selected predictions in `outputs/benchmark`. The LightGBM comparison uses the original O’Neil holdouts and the same Morgan fingerprints and expression features.

```bash
python -m pip install 'matplotlib>=3.8' scipy lightgbm==4.7.0
python -m src.eval.dose_response_cases
python -m src.eval.synergy_ranking
python -m src.train.lightgbm_comparison --threads 10
python -m src.eval.lightgbm_comparison --compare-mlp
```

- **Dose–response surfaces and error decomposition.** Selects surfaces nearest the within-group median of the paired squared-error differences and decomposes the NCI-ALMANAC centre error reduction by doxorubicin status. The script exports SVG, PDF and 600 dpi PNG figures under `outputs/dose_response_followup/surface_cases/`. The figures use Times New Roman when it is installed.
- **Synergy ranking after centre completion.** Replaces one held-out centre response per NCI-ALMANAC surface and ranks drug pairs by the mean ComboScore across cell lines. Outputs include per-initialisation ranking metrics and sensitivity comparisons using archived scores, grid means and removal of doxorubicin pairs.
- **O’Neil LightGBM comparison.** Compares direct log-dose, fixed sinusoidal features and fixed random Fourier features in the centre and second-drug high-dose holdouts. Each encoding uses the same two leaf-count settings, with selection by validation RMSE and a 200-round limit. The optional `--compare-mlp` flag adds the original multilayer perceptron results. Use `--smoke` on the training command to check the pipeline on a small subset in a separate output directory.

The defaults write all generated results and models to `outputs/dose_response_followup/`. Each command supports `--help` for input and output paths.
