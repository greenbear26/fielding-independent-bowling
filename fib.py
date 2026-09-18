"""Fielding Independent Bowling (FIB).

Applies a coefficient vector to per-bowler-season stats and reports how well the
resulting FIB predicts next season's economy rate, compared against how well the
current season's economy rate predicts it.

The intended workflow is to try a coefficient vector, read the correlation, and
repeat:

    from fib import evaluate_fib
    evaluate_fib([6.1821, 5.7628, 1.9969, 2.8617, -2.5032, -4.5166, 1.4290, -1.4063, -11.2354])

This module never writes to disk.
"""

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
    # 2015+ Training
    # 6.1821,   # intercept
    # 5.7628,   # Sixes
    # 1.9969,   # Wides
    # 2.8617,   # NoBalls
    # -2.5032,  # BowledWickets
    # -4.5166,  # LbwWickets
    # 1.4290,   # HitWicketWickets
    # -1.4063,  # CaughtAndBowledWickets
    # -11.2354,  # StumpedWickets

    # Total data training
    6.093429067288927,
    6.09152908,   1.73365948,   1.76918265,  -2.28802524,
         -4.20541236,   4.44205485,  -2.71771511, -11.97673927
]

DEFAULT_CSV_PATH = 'data/bowler-stats.csv'
DEFAULT_MIN_YEAR = 2015


def load_bowler_stats(csv_path=DEFAULT_CSV_PATH, min_year=DEFAULT_MIN_YEAR):
    """Load per-bowler-season stats, keeping only seasons from `min_year` onward.

    The CSV holds every season back to 2009, but the model was fitted on 2015+.
    """
    bowler_df = pd.read_csv(csv_path)
    return bowler_df[bowler_df['Year'] >= min_year].copy()


def compute_raw_fib(df, coefficients):
    """Add a `raw_FIB` column: the coefficients applied to per-over feature rates.

    `coefficients[0]` is the intercept; the rest line up with FEATURE_COLUMNS in
    order. raw_FIB lands on an arbitrary scale -- see `normalize_fib`.
    """
    expected = len(FEATURE_COLUMNS) + 1
    if len(coefficients) != expected:
        raise ValueError(
            f'Expected {expected} coefficients (1 intercept + '
            f'{len(FEATURE_COLUMNS)} for {FEATURE_COLUMNS}), got {len(coefficients)}.'
        )

    coefficients = np.asarray(coefficients, dtype=float)
    per_over_rates = df[FEATURE_COLUMNS].div(df['Overs'], axis=0)

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


def build_next_year_pairs(df):
    """Attach each season's following-season economy as `next_year_economy`.

    Only keeps bowler-seasons where that same bowler also has a qualifying season
    the year after, so rows without a follow-up are dropped.
    """
    following = df[['Bowler', 'Year', 'Economy']].rename(columns={'Economy': 'next_year_economy'})
    following['Year'] -= 1
    return df.merge(following, on=['Bowler', 'Year'], how='inner').reset_index(drop=True)


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
        sns.scatterplot(x=column, y='next_year_economy', data=df)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Next Year Economy')
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')

    plt.show()


def evaluate_fib(coefficients=DEFAULT_COEFFICIENTS, csv_path=DEFAULT_CSV_PATH,
                 min_year=DEFAULT_MIN_YEAR, plot=False, verbose=False):
    """Score a coefficient vector and print how it compares against raw economy rate.

    Prints the correlation of FIB with next season's economy alongside the
    correlation of the current season's economy with next season's economy. FIB
    is only an improvement if it correlates more strongly.

    Set `verbose` for per-year normalization tables, `plot` for a scatter
    comparison. Returns a dict with both correlations, their difference, and the
    paired DataFrame.
    """
    bowler_df = load_bowler_stats(csv_path, min_year)
    bowler_df = compute_raw_fib(bowler_df, coefficients)
    bowler_df = normalize_fib(bowler_df)

    if verbose:
        _print_yearly_stats(bowler_df)

    paired_df = build_next_year_pairs(bowler_df)
    correlation_fib = paired_df['FIB'].corr(paired_df['next_year_economy'])
    correlation_economy = paired_df['Economy'].corr(paired_df['next_year_economy'])
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


if __name__ == '__main__':
    evaluate_fib()
