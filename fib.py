"""Fielding Independent Bowling (FIB).

Applies a coefficient vector to per-bowler-season stats and reports how well the
resulting FIB predicts next season's economy rate, compared against how well the
current season's economy rate predicts it.

The intended workflow is to try a coefficient vector, read the correlation, and
repeat:

    from fib import evaluate_fib
    evaluate_fib([6.1821, 5.7628, 1.9969, 2.8617, -2.5032, -4.5166, 1.4290, -1.4063, -11.2354])

To try a different feature set, pass `feature_columns` alongside the coefficients
that were fitted on it.

This module never writes to disk.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Features the model is built from, as per-over rates. `caught` dismissals are
# deliberately absent: they depend on a fielder, which is the premise of the stat.
FEATURE_COLUMNS = [
    'Sixes',
    'Wides',
    'NoBalls',
    'BowledWickets',
    'LbwWickets',
    'HitWicketWickets',
    'CaughtAndBowledWickets',
    'StumpedWickets',
]

# Intercept first, then one coefficient per entry in FEATURE_COLUMNS.
# These are the values fitted in main.ipynb on 2015+ seasons.
DEFAULT_COEFFICIENTS = [

    # 6.093429067288927,
    # 6.09152908,   1.73365948,   1.76918265,  -2.28802524,
    #      -4.20541236,   4.44205485,  -2.71771511, -11.97673927
    6.093429067288927,
    243.66116326,  79.74833612,  19.46100916, -36.60840377,
        -25.23247416,   8.88410971, -10.87086046, -59.88369637
]

DEFAULT_CSV_PATH = 'data/bowler-stats.csv'
DEFAULT_MIN_YEAR = 2020

# Guards brute_force_fib against accidentally launching a multi-hour grid.
DEFAULT_MAX_COMBINATIONS = 10_000_000

# Attached by main.ipynb's aggregation stage, not computed here.
NEXT_YEAR_COLUMN = 'next_year_economy'


def load_bowler_stats(csv_path=DEFAULT_CSV_PATH, min_year=DEFAULT_MIN_YEAR):
    """Load per-bowler-season stats, keeping only seasons from `min_year` onward.

    The CSV holds every season back to 2009, but the model was fitted on 2015+.
    """
    bowler_df = pd.read_csv(csv_path)
    if NEXT_YEAR_COLUMN not in bowler_df.columns:
        raise KeyError(
            f"{csv_path} has no '{NEXT_YEAR_COLUMN}' column -- re-run main.ipynb "
            'through the aggregation stage to regenerate it.'
        )
    return bowler_df[bowler_df['Year'] >= min_year].copy()


def compute_raw_fib(df, coefficients, feature_columns=FEATURE_COLUMNS):
    """Add a `raw_FIB` column: the coefficients applied to per-over feature rates.

    `coefficients[0]` is the intercept; the rest line up positionally with
    `feature_columns`, which defaults to FEATURE_COLUMNS. Pass the two together --
    a vector fitted on one feature set means nothing against another.
    raw_FIB lands on an arbitrary scale -- see `normalize_fib`.
    """
    expected = len(feature_columns) + 1
    if len(coefficients) != expected:
        raise ValueError(
            f'Expected {expected} coefficients (1 intercept + '
            f'{len(feature_columns)} for {feature_columns}), got {len(coefficients)}.'
        )

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise KeyError(f'Feature columns missing from the data: {missing}.')

    coefficients = np.asarray(coefficients, dtype=float)
    per_over_rates = df[feature_columns].div(df['Overs'], axis=0)

    df = df.copy()
    df['raw_FIB'] = coefficients[0] + per_over_rates @ coefficients[1:]
    return df


def normalize_fib(df):
    """Add a `FIB` column: `raw_FIB` rescaled into economy-rate units.

    Done per year, so FIB matches that season's economy mean and standard
    deviation. This absorbs league-wide scoring inflation and makes FIB directly
    comparable to Economy within a season.
    """
    by_year = df.groupby('Year')
    z_score = (df['raw_FIB'] - by_year['raw_FIB'].transform('mean')) / by_year['raw_FIB'].transform('std')

    df = df.copy()
    df['FIB'] = z_score * by_year['Economy'].transform('std') + by_year['Economy'].transform('mean')
    return df


def take_paired_seasons(df):
    """Keep only bowler-seasons that have a following season to be scored against.

    The pairing itself happens in main.ipynb, which writes `next_year_economy`
    into the CSV; it is blank where that bowler has no qualifying season the year
    after, so those rows drop out here.
    """
    df = df[df[NEXT_YEAR_COLUMN].notna()]
    return df.reset_index(drop=True)


def _print_yearly_stats(df):
    """Print per-year mean and standard deviation of Economy, raw_FIB and FIB.

    A sanity check on normalization: FIB's mean and std should equal Economy's
    exactly, by construction.
    """
    for label, how in (('Mean', 'mean'), ('Standard deviation', 'std')):
        stats = df.groupby('Year').agg({'Economy': how, 'raw_FIB': how, 'FIB': how}).reset_index()
        print(f'\n{label} by year:')
        print(stats.to_string(index=False))


def _plot_comparison(df, correlation_fib, correlation_economy):
    """Scatter FIB and current-year Economy against next year's economy, side by side."""
    plt.figure(figsize=(12, 6))

    for position, (column, title, correlation) in enumerate((
        ('FIB', 'FIB vs Next Year Economy', correlation_fib),
        ('Economy', 'Current Year Economy vs Next Year Economy', correlation_economy),
    ), start=1):
        plt.subplot(1, 2, position)
        sns.scatterplot(x=column, y=NEXT_YEAR_COLUMN, data=df)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Next Year Economy')
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')

    plt.show()


def evaluate_fib(coefficients=DEFAULT_COEFFICIENTS, csv_path=DEFAULT_CSV_PATH,
                 min_year=DEFAULT_MIN_YEAR, feature_columns=FEATURE_COLUMNS,
                 plot=False, verbose=False):
    """Score a coefficient vector and print how it compares against raw economy rate.

    Prints the correlation of FIB with next season's economy alongside the
    correlation of the current season's economy with next season's economy. FIB
    is only an improvement if it correlates more strongly.

    `feature_columns` selects which per-over rates the coefficients apply to, so a
    different feature set can be tried without editing this module -- pass the
    same list the coefficients were fitted on.

    Set `verbose` for per-year normalization tables, `plot` for a scatter
    comparison. Returns a dict with both correlations, their difference, and the
    paired DataFrame.
    """
    bowler_df = load_bowler_stats(csv_path, min_year)
    bowler_df = compute_raw_fib(bowler_df, coefficients, feature_columns)
    bowler_df = normalize_fib(bowler_df)

    if verbose:
        _print_yearly_stats(bowler_df)

    paired_df = take_paired_seasons(bowler_df)
    correlation_fib = paired_df['FIB'].corr(paired_df[NEXT_YEAR_COLUMN])
    correlation_economy = paired_df['Economy'].corr(paired_df[NEXT_YEAR_COLUMN])
    improvement = correlation_fib - correlation_economy

    print(f'\nSeasons {min_year}+: {len(bowler_df)} bowler-seasons, '
          f'{len(paired_df)} with a following season.')
    print(f'  FIB      vs next year economy: {correlation_fib:.4f}')
    print(f'  Economy  vs next year economy: {correlation_economy:.4f}')
    print(f'  Improvement over economy rate: {improvement:+.4f} '
          f'({"better" if improvement > 0 else "worse"})')

    if plot:
        _plot_comparison(paired_df, correlation_fib, correlation_economy)

    return {
        'correlation_fib': correlation_fib,
        'correlation_economy': correlation_economy,
        'improvement': improvement,
        'paired_df': paired_df,
    }


def _score_coefficient_batch(per_over_rates, year_rows, economy_by_year, paired_mask, next_year,
                             coefficient_batch):
    """Correlation with next year's economy for each row of `coefficient_batch`.

    A vectorized equivalent of compute_raw_fib -> normalize_fib -> take_paired_seasons
    -> corr, so thousands of vectors can be scored without a pandas pass for each.
    """
    raw_fib = coefficient_batch[:, 0] + per_over_rates @ coefficient_batch[:, 1:].T

    fib = np.empty_like(raw_fib)
    for year, rows in year_rows.items():
        economy_mean, economy_std = economy_by_year[year]
        year_raw = raw_fib[rows]
        z_score = (year_raw - year_raw.mean(axis=0)) / year_raw.std(axis=0, ddof=1)
        fib[rows] = z_score * economy_std + economy_mean

    fib_centered = fib[paired_mask] - fib[paired_mask].mean(axis=0)
    next_centered = next_year - next_year.mean()
    return (next_centered @ fib_centered) / np.sqrt(
        (fib_centered ** 2).sum(axis=0) * (next_centered ** 2).sum()
    )


def brute_force_fib(coefficients=DEFAULT_COEFFICIENTS, span=1.0, step=0.5, indices=None,
                    csv_path=DEFAULT_CSV_PATH, min_year=DEFAULT_MIN_YEAR,
                    feature_columns=FEATURE_COLUMNS, top=10,
                    max_combinations=DEFAULT_MAX_COMBINATIONS, batch_size=5000):
    """Grid-search coefficient vectors around `coefficients`, scoring each like evaluate_fib.

    Each coefficient in `indices` is tried at every value from `base - span` to
    `base + span` in increments of `step`; every combination across those indices is
    scored, with the remaining coefficients held at their base values. `span` and
    `step` may be scalars or one value per entry in `indices`.

    `indices` defaults to every feature coefficient but not the intercept (index 0):
    per-year normalization z-scores raw_FIB, so the intercept cancels out and cannot
    change the correlation. For the same reason, scaling every feature coefficient by
    one positive factor leaves the correlation unchanged.

    The grid size is the product of the per-index value counts and grows quickly;
    a search over `max_combinations` raises instead of running.

    Prints the `top` best vectors. Returns a DataFrame of every vector tried, sorted
    by `correlation_fib` descending, with its `improvement` over economy rate.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    expected = len(feature_columns) + 1
    if len(coefficients) != expected:
        raise ValueError(
            f'Expected {expected} coefficients (1 intercept + '
            f'{len(feature_columns)} for {feature_columns}), got {len(coefficients)}.'
        )

    indices = list(range(1, expected)) if indices is None else list(indices)
    spans = np.broadcast_to(np.asarray(span, dtype=float), (len(indices),))
    steps = np.broadcast_to(np.asarray(step, dtype=float), (len(indices),))
    if np.any(steps <= 0):
        raise ValueError('`step` must be positive.')

    # linspace rather than arange so floating-point drift can't drop the endpoint.
    value_grids = [
        coefficients[index] + np.linspace(-index_span, index_span, int(round(2 * index_span / index_step)) + 1)
        for index, index_span, index_step in zip(indices, spans, steps)
    ]
    total = int(np.prod([len(values) for values in value_grids]))
    if total > max_combinations:
        raise ValueError(
            f'Grid has {total:,} combinations, over max_combinations={max_combinations:,}. '
            'Narrow `indices`, shrink `span`, widen `step`, or raise `max_combinations`.'
        )

    bowler_df = load_bowler_stats(csv_path, min_year).reset_index(drop=True)
    missing = [column for column in feature_columns if column not in bowler_df.columns]
    if missing:
        raise KeyError(f'Feature columns missing from the data: {missing}.')

    per_over_rates = bowler_df[feature_columns].div(bowler_df['Overs'], axis=0).to_numpy()
    year_rows = bowler_df.groupby('Year').indices
    economy_by_year = {
        year: (bowler_df['Economy'].iloc[rows].mean(), bowler_df['Economy'].iloc[rows].std())
        for year, rows in year_rows.items()
    }
    paired_mask = bowler_df[NEXT_YEAR_COLUMN].notna().to_numpy()
    next_year = bowler_df.loc[paired_mask, NEXT_YEAR_COLUMN].to_numpy()

    paired_df = take_paired_seasons(bowler_df)
    correlation_economy = paired_df['Economy'].corr(paired_df[NEXT_YEAR_COLUMN])

    print(f'\nSearching {total:,} coefficient vectors over indices {indices} '
          f'(seasons {min_year}+, {paired_mask.sum()} with a following season).')

    combinations = itertools.product(*value_grids)
    batches = []
    while batch := list(itertools.islice(combinations, batch_size)):
        coefficient_batch = np.tile(coefficients, (len(batch), 1))
        coefficient_batch[:, indices] = batch
        correlations = _score_coefficient_batch(per_over_rates, year_rows, economy_by_year,
                                                paired_mask, next_year, coefficient_batch)
        batches.append(np.column_stack([coefficient_batch, correlations]))

    results = pd.DataFrame(np.vstack(batches),
                           columns=['Intercept', *feature_columns, 'correlation_fib'])
    results['improvement'] = results['correlation_fib'] - correlation_economy
    results = results.sort_values('correlation_fib', ascending=False).reset_index(drop=True)

    print(f'  Economy vs next year economy: {correlation_economy:.4f}')
    print(f'  Best FIB vs next year economy: {results["correlation_fib"].iloc[0]:.4f} '
          f'({results["improvement"].iloc[0]:+.4f})')
    print(f'\nTop {top}:')
    print(results.head(top).to_string())

    return results


if __name__ == '__main__':
    evaluate_fib()
