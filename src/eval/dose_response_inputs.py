"""Shared inputs for the supplementary dose-response analyses."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics_and_ranks import selected_predictions
from src.splits.make_holdout_splits import input_digest as sha256

ALMANAC = 'almanac_dm47_batch_checked'
ONEIL = 'oneil_batch_checked'
METHODS = {'raw64_fixed': 'Direct log-dose', 'linear64_mlp': 'Linear projection + MLP',
           'D3': 'Random Fourier features + MLP'}
SEEDS = range(3)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def input_paths(description, output, *, predictions=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--split-dir', type=Path, default=Path('outputs/splits'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/dose_response_followup') / output)
    if predictions:
        parser.add_argument('--runs', type=Path, default=Path('outputs/benchmark'))
    return parser


def holdout_path(split_dir, dataset, task):
    return split_dir / 'original' / dataset / task / 'split_indices.npz'


def load_holdout(data_file, split_dir, dataset, task):
    path = holdout_path(split_dir, dataset, task)
    metadata = load_json(path.with_name('split_metadata.json'))
    if (metadata['input_sha256'] != sha256(data_file) or metadata['indices_sha256'] != sha256(path)
            or metadata['task'] != task or metadata['pair_seed'] != 0 or metadata['validation_fraction'] != .25):
        raise ValueError(f'Split and processed dataset do not match the original benchmark: {path}')
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in ('train', 'val', 'test')}


def prediction_manifest(runs, dataset, task, method):
    return runs / 'original' / dataset / task / 'joint_concatenation' / method / 'selected.json'


def matched_predictions(grid, test_rows, runs, dataset, task, methods):
    rows = np.sort(test_rows)
    held = grid.iloc[rows][['PAIR', 'CELLNAME', 'CONC1_LOG', 'CONC2_LOG', 'PERCENTGROWTH']].copy()
    held = held.rename(columns={'PERCENTGROWTH': 'observed'})
    for method in methods:
        manifest = prediction_manifest(runs, dataset, task, method)
        paths = selected_predictions(manifest)
        if set(paths) != set(SEEDS):
            raise ValueError(f'Expected three original initialisations in {manifest}')
        for seed, path in sorted(paths.items()):
            with np.load(path, allow_pickle=False) as data:
                order = np.argsort(data['row_index'])
                np.testing.assert_array_equal(data['row_index'][order], rows)
                for column, key in [('observed', 'y_true'), ('CONC1_LOG', 'conc1_log'), ('CONC2_LOG', 'conc2_log')]:
                    np.testing.assert_allclose(held[column], data[key][order], rtol=1e-6, atol=1e-5)
                held[f'{method}_initialization_{seed}'] = data['y_pred'][order]
    held.insert(0, 'row_index', rows)
    return held


def almanac_centre(data_dir, split_dir, runs, columns=()):
    source = data_dir / ALMANAC / 'pairs.parquet'
    grid = pd.read_parquet(source, columns=['PAIR', 'CELLNAME', 'CONC1_LOG', 'CONC2_LOG', 'PERCENTGROWTH', *columns])
    rows = load_holdout(source, split_dir, ALMANAC, 'centre')['test']
    pred = matched_predictions(grid, rows, runs, ALMANAC, 'centre', METHODS)
    if pred.duplicated(['PAIR', 'CELLNAME']).any() or len(pred) != 175566:
        raise ValueError('Unexpected NCI-ALMANAC centre-test coverage')
    return grid, pred


def doxorubicin(pairs):
    return pairs.str.contains(r'(?:^|_)123127(?:_|$)', regex=True)
