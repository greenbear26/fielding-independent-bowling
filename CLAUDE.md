# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Builds "FIB" (Fielding Independent Bowling), a cricket statistic intended to predict a bowler's
*next-season* economy rate better than their *current* economy rate does, by using only outcomes a
bowler controls without help from fielders. Modeled on baseball's FIP. Data is IPL ball-by-ball
data from https://github.com/ritesh-ojha/IPL-DATASET.

Two files do the work. [main.ipynb](main.ipynb) handles data cleaning, per-bowler-season
aggregation, pairing each season with the bowler's next one, feature engineering, and fitting the
regression; it then scores and grid-searches the fit by calling into `fib.py`.
[fib.py](fib.py) takes a coefficient vector and scores it, or grid-searches around one. There is no
package, no test suite, and no lint config.

The project is mid-experiment and the notebook is edited often: features are switched on and off by
commenting lines out, and whole cells (such as scaling) are toggled the same way. Read the current
cell contents rather than assuming a past configuration.

## Environment

`.venv/` is a plain `python -m venv` (Python 3.13.7, Homebrew) and is git-ignored via `.venv/.gitignore`.
There is **no requirements.txt or pyproject.toml**. Dependencies were pip-installed ad hoc. Currently:
pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, matplotlib 3.11.1, seaborn 0.13.2, ipykernel/jupyter.
If you add an import, install it into `.venv` and mention it to the user; consider proposing a
requirements file rather than silently creating one. `nbconvert`/`nbclient` are **not** installed,
so the notebook can't be executed headlessly. The user runs it in VS Code with the saved kernel
`.venv (3.13.7)`.

```bash
source .venv/bin/activate
python -c "from fib import evaluate_fib; evaluate_fib([...], feature_columns=[...], min_year=2000)"
```

`python fib.py` (which calls `evaluate_fib()` with defaults) **currently raises `KeyError`**: see
"Stale defaults" below. Always pass `feature_columns` explicitly.

## Pipeline

Cell numbers below are 0-indexed and drift as cells are added or removed. Each code cell starts with
a one-line comment, so find a cell by that comment if the numbers have moved. The notebook is a
linear chain of in-memory DataFrames and must be run top to bottom. `fib.py` is independent: it
reads `data/bowler-stats.csv` from disk, so it only needs the notebook to have been run at some
earlier point.

1. **Load & clean** (cells 2–10) → `filtered_df`
   Merges `data/ball-data.csv` with `data/match-info.csv` on `ID` (renamed from `match_number`),
   drops identity/result columns, sorts by `match_date, Innings, Overs, BallNumber`, adds `Year`.
2. **Aggregate per bowler-season** (cell 11) → `bowler_df`
   Nested loop over `Year` × `Bowler`. **Bowler-seasons under 100 legal balls are skipped.**
3. **Pair with the following season** (cell 12) → `next_year_economy` column.
   Looks up the same `Bowler` at `Year + 1`; blank where there is no qualifying season the year
   after (467 of 1072 rows). This is the target `fib.py` scores against.
4. **Scaling** (cell 14), **currently commented out.** When enabled it copies the frame to
   `unscaled_bowler_df` and min-max scales the count columns to 0–1 across all years. While it's
   off, `bowler_df` and therefore `bowler-stats.csv` hold **raw counts**.
5. **Derived columns** (cell 15): `_Squared`/`_Cubed`/`_4` powers of `DotBalls`, `Sixes`, `Wides`,
   `NoBalls`, `FieldingIndependentWickets`, plus `Era` (0 before 2014, 1 through 2022, 2 after).
6. **Save** (cell 16, the only write) → `data/bowler-stats.csv`. The line writing
   `bowler-stats-unscaled.csv` is commented out, so that committed file is not regenerated.
7. **Fit** (cells 18–19) → `fib_reg`. Cell 18 defines `feature_columns` and `min_year` (currently
   2000, i.e. every season), then fits a `LinearRegression` of `feature / Overs` against `Economy`.
8. **Score** (cell 21): `evaluate_fib` with the fitted vector and the same `feature_columns`/`min_year`.
9. **Brute force** (cell 23): `brute_force_fib` around `best_coefficients`, which is **hard-coded**
   and overrides the fitted vector. Re-fitting in cell 18 does not change what gets searched.
   Cell 24 records the best result found so far.
10. **Plot** (cell 26) scatters each `feature / Overs` against next year's economy.

Cell 27 is the running "What I did / Next steps" log.

### Scoring a coefficient vector (`fib.py`)

`evaluate_fib(coefficients, feature_columns=..., min_year=...)` applies a vector to
`feature / Overs` rates, normalizes, drops seasons with no following season
(`take_paired_seasons`, which only filters on `next_year_economy`), and prints
`corr(FIB, next_year_economy)` against `corr(Economy, next_year_economy)`. **That comparison is the
project's success metric**: FIB is only an improvement if it beats plain economy rate.

Element 0 of the vector is the intercept; the rest align positionally with `feature_columns`.
`compute_raw_fib` raises on a length mismatch or a missing column, but **reordering features still
silently changes what a vector means**. Coefficients are only meaningful against the scaling state
they were fitted under: a vector fitted on 0–1 scaled data is nonsense against raw counts and vice versa.

`brute_force_fib(coefficients, span, step, indices=..., feature_columns=..., min_year=...)`
grid-searches every combination of `base ± span` in `step` increments for the coefficients in
`indices` (default: all features, not the intercept), and returns every vector sorted by
correlation. It reimplements scoring in numpy (`_score_coefficient_batch`), so **any change to
`normalize_fib` or `take_paired_seasons` must be mirrored there**. To catch drift, check that
`brute_force_fib(span=0, ...)` reproduces `evaluate_fib`'s correlation. It raises above
`max_combinations` (default 10,000,000). Two things never change the correlation, so grid
points that vary only them are redundant: the intercept (per-year z-scoring cancels it) and
multiplying all feature coefficients by one positive factor.

Nothing in `fib.py` writes to disk.

### Normalization

`raw_FIB` is on an arbitrary scale, so `normalize_fib` rescales it per year to the same mean and
standard deviation as that year's `Economy`, producing `FIB` in economy-rate units. This absorbs
league-wide scoring inflation. As a result, `FIB`'s per-year mean and std always equal
`Economy`'s exactly, so the `verbose=True` tables matching is a tautology, not evidence of a good model.

### Stale defaults in `fib.py`

`FEATURE_COLUMNS` (`Sixes, Wides, NoBalls, BowledWickets, LbwWickets, HitWicketWickets,
CaughtAndBowledWickets, StumpedWickets`) and `DEFAULT_COEFFICIENTS` date from an earlier aggregation
that split wickets by type and from 0–1 scaled data. The notebook no longer produces the split wicket
columns, so the defaults fail against the current CSV. `DEFAULT_MIN_YEAR` is 2020 while the
notebook uses 2000. The module docstring example is also stale. Don't "fix" these without
asking which feature set the user wants as the default.

## Domain rules encoded in the aggregation (cell 11)

These cricket-specific decisions are easy to break:

- **Legal balls**: total rows minus deliveries whose `ExtraType` is one of `'wides'`, `'noballs'`,
  `'penalty,wides'`, `'byes,noballs'`, `'legbyes,noballs'`, `'penalty'`. `Overs = legal balls / 6`.
- **Runs charged to the bowler**: `BatsmanRun` + wide runs + no-ball runs. Byes and leg-byes are
  not charged. For `penalty,wides` the 5-run penalty is subtracted (`max(0, sum - 5)`), and
  `byes,noballs` / `legbyes,noballs` are charged 1 run each.
- **`Wides` and `NoBalls` are runs conceded, not delivery counts.** Note that `wide_runs` sums
  `ExtrasRun` over both `wides` and `noballs` rows, so no-ball extras land in `Wides` as well as
  `NoBalls`, and are therefore counted twice in `TotalRuns`/`Economy`. This looks unintentional
  but hasn't been confirmed with the author; don't change it silently.
- **`DotBalls`** counts deliveries with `BatsmanRun == 0`, which includes wides, no-balls and byes.
- **`TotalWickets`**: `bowled, caught, lbw, stumped, hit wicket, caught and bowled` (run-outs and
  retirements excluded).
- **`FieldingIndependentWickets`** = bowled + lbw + stumped. Hit-wicket and caught-and-bowled are
  computed but left out (alternative sum is commented out beside it). `caught` is excluded on
  purpose, because it depends on a fielder, which is the premise of the stat. `stumped` is a
  judgment call the author flagged as uncertain.
- **Known data gap**: overthrows aren't in the dataset and are ignored.

## Data files

| File | Role |
| --- | --- |
| `ball-data.csv` | Input, ~295k deliveries, key `ID, Innings, Overs, BallNumber` |
| `match-info.csv` | Input, ~1.2k matches, `match_number` → renamed `ID` |
| `bowler-stats.csv` | Written by cell 16; 1072 bowler-seasons 2008–2026, key `Year, Bowler`. Scaled or raw depending on cell 14 (currently raw) |
| `bowler-stats-unscaled.csv` | Committed but no longer written; may not match the current aggregation |

`Bowler` names are the dataset's own initials-style strings (e.g. `Z Khan`); the naming scheme is
inconsistent but unique per player. Join on names only within this dataset.

## Conventions

- Notebook aggregation and pairing (cells 11–12) use explicit Python loops (`iterrows`, nested
  `for`). Match that style rather than vectorizing unless asked. `fib.py` is vectorized throughout;
  keep it that way, since it is re-run repeatedly during tuning.
- `fib.py` functions copy their input DataFrame before adding columns (`df.copy()`), so they can be
  chained or called out of order safely.
- Cell comments are terse one-liners above each block. Commented-out lines are deliberate toggles
  and records of past experiments, not dead code. Markdown cells hold open questions and "MAYBE"
  notes. Keep both intact; they are the project's design log.
