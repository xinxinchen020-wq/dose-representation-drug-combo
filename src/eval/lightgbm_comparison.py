"""Validate LightGBM outputs and report where the high-dose test lies relative to training support."""

import numpy as np
import pandas as pd

from src.eval.dose_response_inputs import ONEIL, holdout_path, input_paths, load_holdout, load_json, prediction_manifest, sha256, write_json
from src.eval.metrics_and_ranks import selected_predictions
from src.train.lightgbm_comparison import ENCODINGS, TASKS


def dose_support(train, test):
    max_by_surface = train.groupby(['PAIR', 'CELLNAME']).CONC2_LOG.max()
    max_by_drug = train.groupby('DRUG2_IDX').CONC2_LOG.max()
    max_either_slot = pd.concat([train[['DRUG2_IDX', 'CONC2_LOG']],
                                 train[['DRUG1_IDX', 'CONC1_LOG']].set_axis(['DRUG2_IDX', 'CONC2_LOG'], axis=1)])
    max_either_slot = max_either_slot.groupby('DRUG2_IDX').CONC2_LOG.max()
    surface = max_by_surface.reindex(pd.MultiIndex.from_frame(test[['PAIR', 'CELLNAME']])).to_numpy()
    return {
        'above_same_pair_and_cell': test.CONC2_LOG.to_numpy() > surface,
        'above_same_drug_in_slot2': test.CONC2_LOG.to_numpy() > max_by_drug.reindex(test.DRUG2_IDX).to_numpy(),
        'above_same_drug_in_either_slot': test.CONC2_LOG.to_numpy() > max_either_slot.reindex(test.DRUG2_IDX).to_numpy(),
        'above_global_slot2': test.CONC2_LOG.to_numpy() > train.CONC2_LOG.max(),
    }


def paired_intervals(task, errors):
    """Resample the same test drug pairs in each encoding, holding each fitted model fixed."""
    per_pair = errors.groupby('PAIR', sort=True).sum(numeric_only=True)
    draw = np.random.default_rng(20261116).integers(len(per_pair), size=(4000, len(per_pair)))
    direct = per_pair['direct_log_dose'].to_numpy()[draw].sum(axis=1)
    results = []
    for encoding in ENCODINGS:
        if encoding == 'direct_log_dose' or encoding not in per_pair:
            continue
        total = per_pair[encoding].to_numpy()
        offset = np.sqrt(total[draw].sum(axis=1) / direct) - 1
        lower, upper = np.quantile(offset, [.025, .975])
        results.append(dict(task=task, encoding=ENCODINGS[encoding], test_drug_pairs=len(per_pair), resamples=len(draw),
                            relative_rmse_change=float(np.sqrt(total.sum() / per_pair.direct_log_dose.sum()) - 1),
                            lower=float(lower), upper=float(upper)))
    return results


def main():
    parser = input_paths(__doc__, 'lightgbm', predictions=True)
    parser.add_argument('--compare-mlp', action='store_true', help='Also read the original validation-selected fits.')
    args = parser.parse_args()
    data_file = args.data_dir / ONEIL / 'pairs.parquet'
    frame = pd.read_parquet(data_file)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    support = {}
    records = []
    missing = []
    subgroup = []
    uncertainty = []
    for task in TASKS:
        split_path = holdout_path(args.split_dir, ONEIL, task)
        split = load_holdout(data_file, args.split_dir, ONEIL, task)
        train = frame.iloc[split['train']]
        test_rows = split['test']
        test = frame.iloc[test_rows]
        masks = dose_support(train, test)
        if task == 'drug2_high_dose' and not masks['above_same_pair_and_cell'].all():
            raise ValueError('A high-dose test point lies within the training interval for its surface')
        support[task] = dict(n_test=len(test), split_sha256=sha256(split_path),
                             counts={label: int(mask.sum()) for label, mask in masks.items()})
        errors = test[['PAIR']].copy()
        for encoding, label in ENCODINGS.items():
            directory = out / 'full' / task / encoding
            if not (directory / 'summary.json').exists():
                missing.append([task, encoding])
                continue
            result = load_json(directory / 'summary.json')
            chosen = min(result['candidate_results'], key=lambda row: (row['validation_rmse'], row['candidate']))
            if chosen['candidate'] != result['selected_candidate']:
                raise ValueError('Selection does not match lowest validation RMSE')
            with np.load(directory / 'test_predictions.npz') as data:
                np.testing.assert_array_equal(data['row_index'], test_rows)
                np.testing.assert_array_equal(data['y_true'], test.PERCENTGROWTH.astype(np.float32))
                error = data['y_pred'] - data['y_true']
            np.testing.assert_allclose(np.sqrt(np.mean(error**2)), result['test_rmse'])
            np.testing.assert_allclose(np.mean(abs(error)), result['test_mae'])
            errors[encoding] = error**2
            records.append(dict(task=task, encoding=label, validation_rmse=result['validation_rmse'], test_rmse=result['test_rmse'],
                                test_mae=result['test_mae'], chosen_num_leaves=chosen['params']['num_leaves'],
                                best_iteration=chosen['best_iteration'], max_rounds=result['max_rounds'],
                                all_candidates_reached_round_cap=all(row['best_iteration'] == result['max_rounds']
                                                                     for row in result['candidate_results']),
                                prediction_sha256=sha256(directory / 'test_predictions.npz'),
                                model_sha256=sha256(directory / 'selected_model.txt')))
            if task == 'drug2_high_dose':
                for range_type in ['above_same_drug_in_slot2', 'above_global_slot2']:
                    for above, mask in [(True, masks[range_type]), (False, ~masks[range_type])]:
                        if mask.any():
                            subgroup.append(dict(encoding=label, range_type=range_type, above=above, points=int(mask.sum()),
                                                 rmse=np.sqrt(np.mean(error[mask]**2)), mae=np.mean(abs(error[mask]))))
        if 'direct_log_dose' in errors:
            uncertainty.extend(paired_intervals(task, errors))
    summary = pd.DataFrame(records)
    if len(summary):
        reference = summary.loc[summary.encoding.eq(ENCODINGS['direct_log_dose'])].set_index('task').test_rmse
        summary['relative_rmse_change_vs_direct'] = summary.test_rmse / summary.task.map(reference) - 1
        summary.to_csv(out / 'comparison.csv', index=False)
        if args.compare_mlp:
            source = []
            for task in TASKS:
                test_rows = load_holdout(data_file, args.split_dir, ONEIL, task)['test']
                expected = frame.iloc[test_rows].PERCENTGROWTH
                reference = None
                for method, label in [('raw64_fixed', 'Direct log-dose'), ('D1', 'Fixed sinusoidal'),
                                      ('D3', 'Random Fourier features + dose MLP')]:
                    paths = selected_predictions(prediction_manifest(args.runs, ONEIL, task, method))
                    if set(paths) != {0, 1, 2}:
                        raise ValueError(f'Expected three original initialisations for {task}, {method}')
                    scores = []
                    for path in paths.values():
                        with np.load(path, allow_pickle=False) as data:
                            np.testing.assert_array_equal(data['row_index'], test_rows)
                            np.testing.assert_allclose(data['y_true'], expected, rtol=1e-6, atol=1e-5)
                            scores.append(np.sqrt(np.mean((data['y_pred'].astype(float) - data['y_true'])**2)))
                    rmse = float(np.mean(scores))
                    if reference is None:
                        reference = rmse
                    source.append(dict(task=task, encoding=label, test_rmse=rmse,
                                       relative_rmse_change_vs_direct=rmse / reference - 1,
                                       architecture='Original multilayer perceptron', initializations=3))
            tree = summary[['task', 'encoding', 'test_rmse', 'relative_rmse_change_vs_direct']].assign(
                architecture='LightGBM', initializations=1)
            pd.concat([pd.DataFrame(source), tree], ignore_index=True).to_csv(out / 'with_original_benchmark.csv', index=False)
        print(summary.drop(columns=['model_sha256', 'prediction_sha256']).to_string(index=False))
    if subgroup:
        pd.DataFrame(subgroup).to_csv(out / 'highest_dose_training_range_subgroups.csv', index=False)
    if uncertainty:
        pd.DataFrame(uncertainty).to_csv(out / 'paired_drug_pair_rmse_intervals.csv', index=False)
    write_json(out / 'training_dose_support.json', support)
    write_json(out / 'completion.json', dict(completed=len(records), expected=len(TASKS) * len(ENCODINGS), missing=missing))


if __name__ == '__main__':
    main()
