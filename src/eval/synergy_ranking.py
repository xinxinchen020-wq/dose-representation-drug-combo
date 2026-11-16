"""Evaluate drug-pair ranking after replacing one measured centre per surface."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src.eval.dose_response_inputs import METHODS, SEEDS, almanac_centre, doxorubicin, input_paths, write_json

KEYS = ['PAIR', 'CELLNAME']


def rank_metrics(pair, reference, predicted):
    error = predicted - reference
    metrics = dict(rmse=np.sqrt(np.mean(error**2)), mae=np.mean(abs(error)),
                   spearman=spearmanr(reference, predicted).statistic,
                   kendall=kendalltau(reference, predicted).statistic)
    for cutoff in [100, int(np.ceil(len(pair) * .05))]:
        actual = set(np.lexsort((pair, -reference))[:cutoff])
        estimated = set(np.lexsort((pair, -predicted))[:cutoff])
        overlap = len(actual & estimated)
        label = 'top100' if cutoff == 100 else 'top5pct'
        metrics.update({f'{label}_k': cutoff, f'{label}_hits': overlap,
                        f'{label}_recall': overlap / cutoff, f'{label}_jaccard': overlap / (2 * cutoff - overlap)})
    return metrics


def main():
    args = input_paths(__doc__, 'synergy_ranking', predictions=True).parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    grid, held = almanac_centre(args.data_dir, args.split_dir, args.runs, ['EXPECTEDGROWTH', 'SCORE'])
    grid['derived'] = grid.EXPECTEDGROWTH.astype(float) - grid.PERCENTGROWTH.astype(float)
    grid['archived'] = grid.SCORE.astype(float)
    grouped = grid.groupby(KEYS, sort=False)
    surface = grouped.agg(n=('PERCENTGROWTH', 'size'), derived_n=('derived', 'count'), archived_n=('archived', 'count'),
                          derived_sum=('derived', 'sum'), archived_sum=('archived', 'sum'))
    held = held.join(surface, on=KEYS, validate='one_to_one')
    valid = held.n.eq(held.derived_n) & held.n.eq(held.archived_n)
    kept = held.loc[valid].copy()
    dropped = held.loc[~valid, [*KEYS, 'row_index', 'n', 'derived_n', 'archived_n']]
    dropped.to_csv(out / 'excluded_surfaces.csv', index=False)
    if (len(kept), kept.PAIR.nunique()) != (175490, 3858):
        raise ValueError('Unexpected complete-surface coverage')
    kept['exact_centre_response'] = grid.PERCENTGROWTH.iloc[kept.row_index].to_numpy(dtype=float)
    kept['doxorubicin_pair'] = doxorubicin(kept.PAIR)
    metrics = []
    pair_exports = []
    surface_errors = []
    decomposition = []
    for rule, reference_column, average in [('expected_minus_observed_sum', 'derived_sum', False),
                                             ('archived_score_sum', 'archived_sum', False),
                                             ('expected_minus_observed_mean', 'derived_sum', True)]:
        scale = kept.n if average else 1
        values = pd.DataFrame({'PAIR': kept.PAIR, 'reference': kept[reference_column] / scale})
        for method, label in METHODS.items():
            for seed in SEEDS:
                name = f'{method}_initialization_{seed}'
                replacement = (kept.exact_centre_response - kept[name]) / scale
                values[name] = values.reference + replacement
                if rule == 'expected_minus_observed_sum':
                    surface_errors.append(dict(method=label, seed=seed, surfaces=len(kept),
                                               rmse=np.sqrt(np.mean(replacement**2)), mae=np.mean(abs(replacement))))
        pairs = values.groupby('PAIR', sort=True).mean()
        pairs.insert(0, 'surfaces', values.groupby('PAIR').size())
        pairs.insert(0, 'score_rule', rule)
        pair_exports.append(pairs.reset_index())
        if rule == 'expected_minus_observed_sum':
            groups = doxorubicin(pairs.index.to_series()).map({True: 'doxorubicin', False: 'other'})
            for method, label in METHODS.items():
                if method == 'raw64_fixed':
                    continue
                for seed in SEEDS:
                    direct = pairs[f'raw64_fixed_initialization_{seed}'] - pairs.reference
                    encoded = pairs[f'{method}_initialization_{seed}'] - pairs.reference
                    contribution = (direct**2 - encoded**2).groupby(groups).sum() / len(pairs)
                    for group, value in contribution.items():
                        decomposition.append(dict(method=label, seed=seed, drug_pair_group=group,
                                                  contribution_to_pair_mse_reduction=value))
        for scope, cutoff_pairs in [('all', pairs), ('excluding_doxorubicin', pairs.loc[~doxorubicin(pairs.index.to_series())])]:
            for method, label in METHODS.items():
                for seed in SEEDS:
                    name = f'{method}_initialization_{seed}'
                    metrics.append(dict(score_rule=rule, scope=scope, method=label, seed=seed, n_pairs=len(cutoff_pairs),
                                        **rank_metrics(cutoff_pairs.index.to_numpy(), cutoff_pairs.reference.to_numpy(),
                                                       cutoff_pairs[name].to_numpy())))
    results = pd.DataFrame(metrics)
    results.to_csv(out / 'metrics_by_initialization.csv', index=False)
    results.groupby(['score_rule', 'scope', 'method'], sort=False).mean(numeric_only=True).drop(columns='seed').to_csv(out / 'summary.csv')
    pd.concat(pair_exports, ignore_index=True).to_csv(out / 'drug_pair_scores.csv', index=False)
    pd.DataFrame(surface_errors).to_csv(out / 'surface_score_error_by_initialization.csv', index=False)
    pd.DataFrame(decomposition).to_csv(out / 'drug_pair_mse_decomposition.csv', index=False)
    write_json(out / 'coverage.json', dict(candidate_surfaces=len(held), eligible_surfaces=len(kept), excluded_surfaces=len(dropped),
               candidate_pairs=held.PAIR.nunique(), eligible_pairs=kept.PAIR.nunique(), doxorubicin_pairs=kept.loc[kept.doxorubicin_pair, 'PAIR'].nunique(),
               excluded_missing_on_heldout=int(grid.EXPECTEDGROWTH.iloc[dropped.row_index].isna().sum()),
               max_abs_derived_vs_archived_point_score=float((grid.derived - grid.archived).abs().max())))
    print(results.loc[(results.score_rule == 'expected_minus_observed_sum') & (results.scope == 'all')].to_string(index=False))


if __name__ == '__main__':
    main()
