"""Fit fixed dose representations with the original O'Neil splits."""

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.eval.dose_response_inputs import ONEIL, holdout_path, input_paths, load_holdout, load_json, sha256, write_json
from src.models.dose_encoders import DoseEncFixedSin, DoseEncFourier

ENCODINGS = {
    'direct_log_dose': 'Direct log-dose',
    'fixed_sinusoidal': 'Fixed sinusoidal',
    'fixed_random_fourier': 'Fixed random Fourier features',
}
TASKS = ['centre', 'drug2_high_dose']
CANDIDATES = [dict(num_leaves=15, min_data_in_leaf=100, learning_rate=0.12),
              dict(num_leaves=31, min_data_in_leaf=100, learning_rate=0.12)]
MAX_ROUNDS = 200
EARLY_STOPPING = 20


def load_inputs(data, split_dir, task):
    source = data / 'pairs.parquet'
    partitions = load_holdout(source, split_dir, ONEIL, task)
    if [len(partition) for partition in partitions.values()] != [244860, 20300, 61320]:
        raise ValueError('Unexpected split size')
    frame = pd.read_parquet(source)
    with np.load(data / 'drug_reprs.npz', allow_pickle=False) as archive:
        drugs = archive['morgan'].astype(np.float32)
    cells = np.load(data / 'cell_expression.npy', allow_pickle=False).astype(np.float32)
    cells = (cells - cells.mean(axis=0)) / (cells.std(axis=0) + 1e-6)
    return frame, partitions, drugs, cells


def dose_features(frame, encoding):
    torch.set_num_threads(1)
    with torch.inference_mode():
        doses = [torch.tensor(frame[name].to_numpy(copy=True), dtype=torch.float32).nan_to_num(nan=-9.0)
                 for name in ['CONC1_LOG', 'CONC2_LOG']]
        if encoding == 'direct_log_dose':
            return torch.stack(doses, dim=1).numpy()
        if encoding == 'fixed_sinusoidal':
            encoder = DoseEncFixedSin(64)
            return torch.cat([encoder(dose) for dose in doses], dim=1).numpy()
        frequencies = DoseEncFourier(d=64, n_freq=32, sigma=0.25, seed=0).B
        phases = [2 * np.pi * dose.unsqueeze(-1) @ frequencies for dose in doses]
        return torch.cat([item for phase in phases for item in [torch.sin(phase), torch.cos(phase)]], dim=1).numpy()


def make_matrix(frame, rows, drugs, cells, dose):
    indices = frame[['DRUG1_IDX', 'DRUG2_IDX', 'CELL_IDX']].to_numpy(dtype=np.int64)
    components = [(drugs, indices[:, 0]), (drugs, indices[:, 1]), (cells, indices[:, 2]), (dose, np.arange(len(frame)))]
    matrix = np.empty((len(rows), sum(values.shape[1] for values, _ in components)), dtype=np.float32)
    for begin in range(0, len(rows), 4096):
        block = rows[begin:begin + 4096]
        offset = 0
        for values, lookup in components:
            end = offset + values.shape[1]
            matrix[begin:begin + len(block), offset:end] = values[lookup[block]]
            offset = end
    return matrix


def run(task, encoding, args):
    threads, smoke = args.threads, args.smoke
    directory = args.output_dir / ('smoke' if smoke else 'full') / task / encoding
    data = args.data_dir / ONEIL
    split_path = holdout_path(args.split_dir, ONEIL, task)
    signature = {name: sha256(data / name) for name in ['pairs.parquet', 'drug_reprs.npz', 'cell_expression.npy']}
    signature.update(split_sha256=sha256(split_path), num_threads=threads,
                     lightgbm_version=lgb.__version__, max_rounds=5 if smoke else MAX_ROUNDS,
                     early_stopping_rounds=EARLY_STOPPING, candidates=CANDIDATES)
    if (directory / 'summary.json').exists():
        if signature != load_json(directory / 'summary.json').get('signature'):
            raise ValueError(f'Existing fit uses different inputs or settings: {directory}')
        print(f'Skipping completed {task}, {encoding}', flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    frame, partitions, drugs, cells = load_inputs(data, args.split_dir, task)
    if smoke:
        rng = np.random.default_rng(0)
        partitions = {name: np.sort(rng.choice(rows, size=2048, replace=False)) for name, rows in partitions.items()}
    dose = dose_features(frame, encoding)
    targets = frame.PERCENTGROWTH.to_numpy(dtype=np.float32)
    params = dict(objective='regression', metric='rmse', device_type='cpu', verbosity=-1,
                  num_threads=threads, deterministic=True, force_col_wise=True, seed=0,
                  feature_fraction=1.0, feature_fraction_bynode=1.0, bagging_fraction=1.0,
                  bagging_freq=0, max_bin=63, feature_pre_filter=False)
    print(f'{task}, {encoding}: building train and validation bins', flush=True)
    train_rows, val_rows, test_rows = partitions.values()
    train = lgb.Dataset(make_matrix(frame, train_rows, drugs, cells, dose), label=targets[train_rows], params=params)
    train.construct()
    valid = lgb.Dataset(make_matrix(frame, val_rows, drugs, cells, dose), label=targets[val_rows], reference=train, params=params)
    valid.construct()
    gc.collect()
    candidates = []
    chosen = None
    for number, config in enumerate(CANDIDATES):
        trace = {}
        model = lgb.train(params | config, train, num_boost_round=5 if smoke else MAX_ROUNDS,
                          valid_sets=[valid], valid_names=['validation'],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.record_evaluation(trace)])
        score = float(model.best_score['validation']['rmse'])
        record = dict(candidate=number + 1, params=config, validation_rmse=score,
                      best_iteration=model.best_iteration, validation_history=trace['validation']['rmse'])
        candidates.append(record)
        print(f'{task}, {encoding}, candidate {number + 1}: validation RMSE {score:.5f} at {model.best_iteration}', flush=True)
        if chosen is None or score < chosen[0]:
            chosen = (score, model, record)
    del train, valid
    gc.collect()
    score, model, record = chosen
    predictions = []
    for begin in range(0, len(test_rows), 4096):
        rows = test_rows[begin:begin + 4096]
        predictions.append(model.predict(make_matrix(frame, rows, drugs, cells, dose), num_threads=threads))
    predicted = np.concatenate(predictions)
    errors = predicted - targets[test_rows]
    np.savez_compressed(directory / 'test_predictions.npz', row_index=test_rows, y_true=targets[test_rows], y_pred=predicted)
    (directory / 'selected_model.txt').write_text(model.model_to_string(), encoding='utf-8')
    write_json(directory / 'summary.json', dict(task=task, encoding=encoding, lightgbm_version=lgb.__version__, signature=signature,
               counts={name: len(rows) for name, rows in partitions.items()}, fixed_params=params,
               max_rounds=5 if smoke else MAX_ROUNDS, early_stopping_rounds=EARLY_STOPPING,
               candidate_results=candidates, selected_candidate=record['candidate'], best_iteration=model.best_iteration,
               validation_rmse=score, test_rmse=float(np.sqrt(np.mean(errors**2))), test_mae=float(np.mean(abs(errors)))))
    print(f'{task}, {encoding}: test RMSE {np.sqrt(np.mean(errors**2)):.5f}', flush=True)


def main():
    parser = input_paths(__doc__, 'lightgbm')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--encoding', choices=ENCODINGS)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for task in [args.task] if args.task else TASKS:
        for encoding in [args.encoding] if args.encoding else ENCODINGS:
            run(task, encoding, args)


if __name__ == '__main__':
    main()
