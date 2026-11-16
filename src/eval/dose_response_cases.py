"""Select median-error surfaces and plot held-out responses and squared error contributions."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

from src.eval.dose_response_inputs import (
    METHODS, ONEIL, SEEDS, almanac_centre, doxorubicin, input_paths, load_holdout,
    matched_predictions, write_json,
)

KEYS = ['PAIR', 'CELLNAME']
DIRECT = 'raw64_fixed'
FOURIER = 'D3'
PALETTE = {
    'direct': '#4D4D4D',
    'linear_mlp': '#3C5488',
    'rff_mlp': '#E64B35',
    'dox_group': '#365883',
    'other_group': '#935329',
    'measured_line': '#7F8B97',
    'measured_dot': '#263238',
    'highlight_box': '#E64B35',
}
METHOD_COLORS = {DIRECT: PALETTE['direct'], 'linear64_mlp': PALETTE['linear_mlp'], FOURIER: PALETTE['rff_mlp']}
GROUP_COLORS = {'Doxorubicin': PALETTE['dox_group'], 'Other pairs': PALETTE['other_group']}
SOFT_SURFACE_CMAP = LinearSegmentedColormap.from_list('soft_surface', ['#2B4162', '#5C769D', '#B8C6D8', '#F4EFE6', '#E9C46A'])
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 10,
    'mathtext.fontset': 'stix', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#66727D', 'axes.linewidth': .8,
    'xtick.color': '#263238', 'ytick.color': '#263238',
    'text.color': '#1C2630', 'axes.labelcolor': '#1C2630',
    'grid.color': '#E4E8EB', 'grid.linewidth': .6, 'legend.frameon': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
})


def add_error(frame):
    for method in [DIRECT, 'linear64_mlp', FOURIER]:
        columns = [f'{method}_initialization_{seed}' for seed in SEEDS]
        if all(column in frame for column in columns):
            frame[f'{method}_mse'] = np.mean((frame[columns].to_numpy(dtype=float) - frame.observed.to_numpy(dtype=float)[:, None])**2, axis=1)
    frame['delta_mse'] = frame[f'{FOURIER}_mse'] - frame[f'{DIRECT}_mse']
    return frame


def load_oneil(data_dir, split_dir, runs):
    source = data_dir / ONEIL / 'pairs.parquet'
    grid = pd.read_parquet(source, columns=[*KEYS, 'CONC1_LOG', 'CONC2_LOG', 'PERCENTGROWTH'])
    rows = load_holdout(source, split_dir, ONEIL, 'drug2_high_dose')['test']
    predictions = matched_predictions(grid, rows, runs, ONEIL, 'drug2_high_dose', [DIRECT, FOURIER])
    return grid, add_error(predictions)


def median_case(surfaces):
    median = surfaces.delta_mse.median()
    chosen = surfaces.assign(distance=(surfaces.delta_mse - median).abs()).sort_values(['distance', *KEYS]).iloc[0]
    return chosen.PAIR, chosen.CELLNAME, float(median), float(chosen.delta_mse)


def case_grid(grid, preds, pair, cell):
    columns = [f'{method}_initialization_{seed}' for method in [DIRECT, FOURIER] for seed in SEEDS]
    measured = grid.loc[grid.PAIR.eq(pair) & grid.CELLNAME.eq(cell), [*KEYS, 'CONC1_LOG', 'CONC2_LOG', 'PERCENTGROWTH']]
    held = preds.loc[preds.PAIR.eq(pair) & preds.CELLNAME.eq(cell), ['CONC1_LOG', 'CONC2_LOG', *columns]]
    curve = measured.merge(held, on=['CONC1_LOG', 'CONC2_LOG'], how='left', validate='one_to_one')
    curve['held_out'] = curve[columns[0]].notna()
    if curve.held_out.sum() != len(held):
        raise ValueError('Missing held-out points in plotted case')
    return curve


def save(fig, name, figures):
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(figures / f'{name}.{extension}', dpi=200, bbox_inches='tight', facecolor='white')
    high_resolution = figures / '600dpi'
    high_resolution.mkdir(exist_ok=True)
    fig.savefig(high_resolution / f'{name}.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def heatmap(ax, curve, norm):
    pivot = curve.pivot(index='CONC1_LOG', columns='CONC2_LOG', values='PERCENTGROWTH')
    x_edges = np.arange(len(pivot.columns) + 1) - .5
    y_edges = np.arange(len(pivot.index) + 1) - .5
    image = ax.pcolormesh(x_edges, y_edges, pivot, cmap=SOFT_SURFACE_CMAP, norm=norm)
    for row in curve.itertuples(index=False):
        x, y = pivot.columns.get_loc(row.CONC2_LOG), pivot.index.get_loc(row.CONC1_LOG)
        rgb = np.asarray(SOFT_SURFACE_CMAP(norm(row.PERCENTGROWTH))[:3])
        luminance = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4) @ [0.2126, 0.7152, 0.0722]
        color = '#1C2630' if luminance > .20 else 'white'
        ax.text(x, y, f'{row.PERCENTGROWTH:.1f}', ha='center', va='center', color=color, fontsize=8)
        if row.held_out:
            ax.add_patch(plt.Rectangle((x - .47, y - .47), .94, .94, fill=False, ec=PALETTE['highlight_box'], lw=1.8))
    ax.set_xticks(range(len(pivot.columns)), [f'{x:.2f}' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f'{x:.2f}' for x in pivot.index])
    ax.set_xlabel('Drug 2 log₁₀(dose / μM)')
    ax.set_ylabel('Drug 1 log₁₀(dose / μM)')
    return image


def profile(ax, curve, dose_axis, outcomes):
    curve = curve.sort_values(dose_axis)
    held = curve.loc[curve.held_out]
    given = curve.loc[~curve.held_out]
    ax.plot(curve[dose_axis], curve.PERCENTGROWTH, color=PALETTE['measured_line'], lw=1, zorder=1)
    ax.scatter(given[dose_axis], given.PERCENTGROWTH, color=PALETTE['measured_dot'], s=24, label='Measured, non-held-out', zorder=3)
    ax.scatter(held[dose_axis], held.PERCENTGROWTH, ec=PALETTE['measured_dot'], fc='white', marker='D', s=35, label='Measured, held out', zorder=4)
    span = np.diff(curve[dose_axis].agg(['min', 'max']))[0]
    for method, offset in zip([DIRECT, FOURIER], [-.035, .035]):
        for seed, jitter in zip(SEEDS, [-.01, 0, .01]):
            ax.scatter(held[dose_axis] + span * (offset + jitter), held[f'{method}_initialization_{seed}'],
                       marker='s' if method == DIRECT else '^', s=26, color=METHOD_COLORS[method],
                       edgecolors='white', linewidths=.35,
                       label=METHODS[method] if seed == 0 else None, zorder=5)
    ax.set_xticks(curve[dose_axis], [f'{x:.2f}' for x in curve[dose_axis]])
    ax.set_xlabel(f'Drug {1 if dose_axis == "CONC1_LOG" else 2} log₁₀(dose / μM)')
    ax.set_ylim(outcomes)
    ax.set_axisbelow(True)
    ax.grid(axis='y')


def plot_almanac(curves, figures):
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.1), layout='constrained')
    response = pd.concat([curve.PERCENTGROWTH for _, curve in curves])
    norm = Normalize(response.min(), response.max())
    for row, (_, curve) in enumerate(curves):
        ax = axes[row]
        image = heatmap(ax[0], curve, norm)
        case = curve.loc[curve.held_out].iloc[0]
        drugs = ' + '.join('Doxorubicin' if token == '123127' else f'NSC {token}' for token in case.PAIR.split('_'))
        ax[0].set_title(f'{drugs}\n{case.CELLNAME}', loc='left', fontweight='bold')
        values = curve.select_dtypes('number').filter(regex='PERCENTGROWTH|initialization').to_numpy()
        bounds = np.nanmin(values) - 6, np.nanmax(values) + 6
        for j, (dose, fixed) in enumerate([('CONC1_LOG', 'CONC2_LOG'), ('CONC2_LOG', 'CONC1_LOG')], 1):
            profile(ax[j], curve.loc[curve[fixed].eq(case[fixed])], dose, bounds)
            ax[j].set_title(f'Drug {3-j} fixed at {case[fixed]:.2f}')
            if j == 1:
                ax[j].set_ylabel('Growth (%)')
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=4, frameon=False)
    colorbar = fig.colorbar(image, ax=axes[:, 0], shrink=.72, label='Measured growth (%)')
    colorbar.solids.set_rasterized(False)
    save(fig, 'almanac_centre_median_surfaces', figures)


def plot_oneil(curve, figures):
    fig = plt.figure(figsize=(10.5, 5.4), layout='constrained')
    grid = fig.add_gridspec(2, 3, width_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    norm = Normalize(curve.PERCENTGROWTH.min(), curve.PERCENTGROWTH.max())
    image = heatmap(ax, curve, norm)
    ax.set_title(f'{curve.PAIR.iloc[0].replace("_", " + ")}\n{curve.CELLNAME.iloc[0]}', loc='left', fontweight='bold')
    colorbar = fig.colorbar(image, ax=ax, shrink=.65, label='Measured viability (%)')
    colorbar.solids.set_rasterized(False)
    values = curve.select_dtypes('number').filter(regex='PERCENTGROWTH|initialization').to_numpy()
    bounds = np.nanmin(values) - 5, np.nanmax(values) + 5
    for i, (value, part) in enumerate(curve.groupby('CONC1_LOG', sort=True)):
        panel = fig.add_subplot(grid[i // 2, i % 2 + 1])
        profile(panel, part, 'CONC2_LOG', bounds)
        panel.set_title(f'Drug 1 fixed at {value:.2f}')
        if i % 2 == 0:
            panel.set_ylabel('Viability (%)')
    handles, labels = panel.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=4, frameon=False)
    save(fig, 'oneil_drug2_highest_dose_median_surface', figures)


def almanac_decomposition(surfaces, out, figures):
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), layout='constrained')
    is_doxo = doxorubicin(surfaces.PAIR)
    for group, subset in [('Doxorubicin', surfaces.loc[is_doxo]), ('Other pairs', surfaces.loc[~is_doxo])]:
        ranks = np.sort(subset.delta_mse.to_numpy())
        axes[0].plot(ranks, np.arange(1, len(ranks) + 1) / len(ranks), label=f'{group} (n={len(ranks):,})',
                     color=GROUP_COLORS[group], lw=1.6, linestyle='-' if group == 'Doxorubicin' else '--')
        for method in ['linear64_mlp', FOURIER]:
            delta = subset[f'{method}_mse'] - subset[f'{DIRECT}_mse']
            records.append(dict(group=group, method=METHODS[method], n_surfaces=len(subset), n_pairs=subset.PAIR.nunique(),
                                fraction_improved=(delta < 0).mean(), q25=delta.quantile(.25), median=delta.median(), q75=delta.quantile(.75),
                                direct_pooled_rmse=np.sqrt(subset[f'{DIRECT}_mse'].mean()), encoded_pooled_rmse=np.sqrt(subset[f'{method}_mse'].mean()),
                                weighted_mse_reduction=-delta.sum() / len(surfaces)))
    stats = pd.DataFrame(records)
    stats['share_of_total_reduction'] = stats.weighted_mse_reduction / stats.groupby('method').weighted_mse_reduction.transform('sum')
    stats.to_csv(out / 'almanac_error_decomposition.csv', index=False)
    axes[0].set(xscale='symlog', xlabel='Fourier minus direct squared error (percentage points²)', ylabel='Cumulative fraction of surfaces')
    ticks = [-10000, -100, -10, 0, 10, 100, 10000]
    axes[0].set_xticks(ticks, [f'{n:,}'.replace('-', '−') for n in ticks], fontsize=9)
    axes[0].axvline(0, color=PALETTE['measured_line'], lw=.8, linestyle=':')
    axes[0].legend(frameon=False, loc='lower right', fontsize=8)
    for label, color in GROUP_COLORS.items():
        subset = stats.loc[stats.group.eq(label)]
        left = np.zeros(len(subset)) if label == 'Doxorubicin' else stats.loc[stats.group.eq('Doxorubicin'), 'weighted_mse_reduction'].to_numpy()
        axes[1].barh(['Linear projection + MLP', 'Random Fourier features + MLP'], subset.weighted_mse_reduction,
                     left=left, color=color, height=.45)
        if label == 'Doxorubicin':
            for i, record in enumerate(subset.itertuples()):
                axes[1].text(record.weighted_mse_reduction / 2, i, f'{record.share_of_total_reduction:.1%}',
                             color='white', va='center', ha='center', fontweight='bold')
    axes[1].set(xlabel='Contribution to MSE reduction (percentage points²)', ylim=(-.5, 1.5))
    for tick, method in zip(axes[1].get_yticklabels(), ['linear64_mlp', FOURIER]):
        tick.set_color(METHOD_COLORS[method])
        tick.set_fontweight('bold')
    axes[1].invert_yaxis()
    save(fig, 'almanac_centre_error_decomposition', figures)


def centre_shape_check(grid, held, out):
    """Compare the measured centre with linear interpolation of the nearest points on each axis."""
    coordinates = ['CONC1_LOG', 'CONC2_LOG']
    centre = grid.iloc[held.row_index][[*KEYS, *coordinates, 'PERCENTGROWTH']].copy()
    centre['delta_mse'] = held.delta_mse.to_numpy()
    for method in [DIRECT, FOURIER]:
        columns = [f'{method}_initialization_{seed}' for seed in SEEDS]
        centre[f'{method}_bias'] = held[columns].mean(axis=1).to_numpy() - centre.PERCENTGROWTH.to_numpy()
    for i, dose in enumerate(coordinates):
        fixed = coordinates[1 - i]
        joint = grid[[*KEYS, *coordinates, 'PERCENTGROWTH']].merge(
            centre[[*KEYS, *coordinates]].rename(columns={dose: 'centre_dose'}), on=[*KEYS, fixed], validate='many_to_one')
        below = joint.loc[joint[dose] < joint.centre_dose].sort_values(dose).drop_duplicates(KEYS, keep='last')
        above = joint.loc[joint[dose] > joint.centre_dose].sort_values(dose).drop_duplicates(KEYS, keep='first')
        neighbors = below.set_index(KEYS).join(above.set_index(KEYS)[[dose, 'PERCENTGROWTH']], rsuffix='_higher')
        weight = (neighbors.centre_dose - neighbors[dose]) / (neighbors[f'{dose}_higher'] - neighbors[dose])
        expected = neighbors.PERCENTGROWTH + weight * (neighbors.PERCENTGROWTH_higher - neighbors.PERCENTGROWTH)
        lookup = pd.MultiIndex.from_frame(centre[KEYS])
        centre[f'axis{i + 1}_centre_offset'] = centre.PERCENTGROWTH - expected.reindex(lookup).to_numpy()
        lower_response = neighbors.PERCENTGROWTH.reindex(lookup).to_numpy()
        centre[f'axis{i + 1}_change_from_lower_dose'] = centre.PERCENTGROWTH - lower_response
    records = []
    for group, subset in [('Doxorubicin', centre.loc[doxorubicin(centre.PAIR)]), ('Other pairs', centre.loc[~doxorubicin(centre.PAIR)])]:
        for axis in [1, 2]:
            offset = f'axis{axis}_centre_offset'
            data = subset.dropna(subset=[offset])
            records.append(dict(group=group, dose_axis=axis, surfaces=len(data), median_centre_offset=data[offset].median(),
                                median_change_from_lower_dose=data[f'axis{axis}_change_from_lower_dose'].median(),
                                fraction_centre_below_local_linear=data[offset].lt(0).mean(),
                                spearman_local_offset_vs_delta_mse=data[offset].corr(data.delta_mse, method='spearman'),
                                median_direct_signed_error=data[f'{DIRECT}_bias'].median(),
                                median_fourier_signed_error=data[f'{FOURIER}_bias'].median()))
    pd.DataFrame(records).to_csv(out / 'almanac_centre_local_shape_check.csv', index=False)


def main():
    args = input_paths(__doc__, 'surface_cases', predictions=True).parse_args()
    out = args.output_dir / 'results'
    figures = args.output_dir / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    almanac, centres = almanac_centre(args.data_dir, args.split_dir, args.runs)
    centres = add_error(centres)
    almanac_decomposition(centres, out, figures)
    centre_shape_check(almanac, centres, out)
    is_doxo = doxorubicin(centres.PAIR)
    oneil, boundary = load_oneil(args.data_dir, args.split_dir, args.runs)
    sizes = boundary.groupby(KEYS, sort=False).size()
    if len(sizes) != 15330 or not sizes.eq(4).all():
        raise ValueError('Unexpected O’Neil boundary coverage')
    averaged = boundary.groupby(KEYS, sort=False)[[f'{DIRECT}_mse', f'{FOURIER}_mse', 'delta_mse']].mean().reset_index()
    measured_max = oneil.groupby(KEYS).CONC2_LOG.max()
    np.testing.assert_array_equal(boundary.set_index(KEYS).join(measured_max.rename('max_dose')).CONC2_LOG,
                                  boundary.set_index(KEYS).join(measured_max.rename('max_dose')).max_dose)
    overall_direct = np.sqrt(boundary[f'{DIRECT}_mse'].mean())
    overall_fourier = np.sqrt(boundary[f'{FOURIER}_mse'].mean())
    write_json(out / 'oneil_boundary_error.json', dict(n_surfaces=len(averaged), n_heldout_points=len(boundary),
               direct_pooled_rmse=float(overall_direct), fourier_pooled_rmse=float(overall_fourier),
               relative_rmse_change=float(overall_fourier / overall_direct - 1),
               fraction_surfaces_with_higher_mse=float(averaged.delta_mse.gt(0).mean()),
               delta_mse_quantiles={str(q): float(averaged.delta_mse.quantile(q)) for q in [.25, .5, .75]}))
    cases = []
    curves = []
    for label, cohort, grid, pred, surfaces in [
        ('Doxorubicin pair', 'almanac_doxorubicin', almanac, centres, centres.loc[is_doxo]),
        ('Other drug pair', 'almanac_other', almanac, centres, centres.loc[~is_doxo]),
        ('O’Neil Drug 2 highest dose', 'oneil_drug2_highest_dose', oneil, boundary, averaged),
    ]:
        pair, cell, median, delta = median_case(surfaces)
        curve = case_grid(grid, pred, pair, cell)
        curve.to_csv(out / f'{cohort}_measured_and_predicted.csv', index=False)
        cases.append(dict(cohort=cohort, pair=pair, cell_line=cell, median_delta_mse=median, selected_delta_mse=delta,
                          held_out_points=int(curve.held_out.sum()), grid_points=len(curve)))
        curves.append((label, curve))
    write_json(out / 'selected_cases.json', cases)
    plot_almanac(curves[:2], figures)
    plot_oneil(curves[2][1], figures)
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == '__main__':
    main()
