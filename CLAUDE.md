# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Builds "FIB" (Fielding Independent Bowling), a cricket statistic intended to predict a bowler's
*next-season* economy rate better than their *current* economy rate does, by using only outcomes a
bowler controls without help from fielders. Modeled on baseball's FIP. Data is IPL ball-by-ball
data from https://github.com/ritesh-ojha/IPL-DATASET.

Two files do the work. [main.ipynb](main.ipynb) handles data cleaning, per-bowler-season
aggregation, pairing each season with the bowler's next one, 0–1 scaling, and fitting the
regression; it then scores the fit by calling into `fib.py`.
[fib.py](fib.py) takes a coefficient vector and scores it, or grid-searches around one. There is no
package, no test suite, and no lint config.

## Environment

`.venv/` is a plain `python -m venv` (Python 3.13.7, Homebrew) and is git-ignored via `.venv/.gitignore`.
There is **no requirements.txt or pyproject.toml** — dependencies were pip-installed ad hoc. Currently:
pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, matplotlib 3.11.1, seaborn 0.13.2, ipykernel/jupyter.
If you add an import, install it into `.venv` and mention it to the user; consider proposing a
requirements file rather than silently creating one. `nbconvert`/`nbclient` are **not** installed,
so the notebook can't be executed headlessly — the user runs it in VS Code.

```bash
source .venv/bin/activate
jupyter lab main.ipynb          # or run cells via the VS Code kernel ".venv (3.13.7)"
python fib.py                   # score the default coefficient vector
```

The notebook's saved kernel is `.venv (3.13.7)`; select that kernel, not a global Python.

## Pipeline

Cell numbers below are 0-indexed. The notebook is a linear chain of in-memory DataFrames and must
be run top to bottom. `fib.py` is independent — it reads `data/bowler-stats.csv` from disk, so it
only needs the notebook to have been run at some earlier point.

1. **Load & clean** (cells 2–10) → `filtered_df`
   Merges `data/ball-data.csv` with `data/match-info.csv` on `ID` (renamed from `match_number`),
   drops identity/result columns, sorts by `match_date, Innings, Overs, BallNumber`, adds `Year`
   derived from `match_date`.
2. **Aggregate per bowler-season** (cell 11) → `bowler_df`
   Nested loop over `Year` × `Bowler`. **Bowler-seasons under 100 legal balls are skipped.**
3. **Pair with the following season** (cell 12) → `next_year_economy` column.
   Loop looking up the same `Bowler` at `Year + 1`; blank where that bowler has no qualifying season
   the year after (467 of 1072 rows). This is the target `fib.py` scores against, and it lives here
   because it's a property of the data, not of the coefficient vector being tuned.
4. **Derived columns & scaling** (cells 13–16)
   Cell 13 adds `Economy_avg` (that year's league mean) and `Economy_diff`. Cell 15 first copies the
   frame to `unscaled_bowler_df`, then min-max scales the count columns and `Economy_avg`/
   `Economy_diff` of `bowler_df` to 0–1 using the min and max **across all years**. `Economy` and
   `next_year_economy` are never scaled. Cell 16 adds `_Squared`/`_Cubed`/`_4` powers and `Era`
   (0 before 2014, 1 through 2022, 2 after) to both frames — in the scaled frame the powers are
   of the scaled values.
5. **Save** (cell 17, the only write) → `data/bowler-stats.csv` (scaled) and
   `data/bowler-stats-unscaled.csv` (raw counts).
6. **Fit** (cells 19–20) → `fib_reg`
   Cell 19 defines `feature_columns` and `min_year`, then fits a `LinearRegression` of
   `feature / Overs` against `Economy` on the scaled frame. Cell 20 prints the coefficients.
7. **Score & plot** (cells 22, 24) — cell 22 calls `evaluate_fib` with the fitted vector and the
   same `feature_columns`/`min_year`; cell 24 scatters each input against next year's economy.

### Scoring a coefficient vector (`fib.py`)

`evaluate_fib(coefficients)` applies a vector, normalizes, drops seasons with no following season
(`take_paired_seasons`, which just filters on the `next_year_economy` column the notebook wrote —
it does **not** compute the pairing), and prints `corr(FIB, next_year_economy)` against
`corr(Economy, next_year_economy)`. **That comparison is the project's success metric**: FIB is only
an improvement if it beats plain economy rate. The intended loop is to try a vector, read the
correlation, and repeat:

```python
from fib import evaluate_fib
evaluate_fib()                                       # DEFAULT_COEFFICIENTS on FEATURE_COLUMNS
evaluate_fib(coefficients, feature_columns=[...])    # any other feature set
```

`plot=True` draws the two-panel comparison scatter; `verbose=True` prints per-year normalization
tables. Nothing in `fib.py` **writes to disk**, so rapid tweaking can't dirty a tracked file.
Defaults: `csv_path='data/bowler-stats.csv'` (the scaled file) and `min_year=2020`.

Element 0 of the vector is the intercept; the rest align positionally with `feature_columns`
(default `FEATURE_COLUMNS`, 8 features). `compute_raw_fib` validates the length and raises, so a
mismatched vector fails loudly rather than silently mis-mapping — but **reordering
`FEATURE_COLUMNS` still silently changes what a given vector means**, since length is unchanged.
`DEFAULT_COEFFICIENTS` were fitted on the **scaled** CSV (hence magnitudes like Sixes ≈ 244); they
are meaningless against `bowler-stats-unscaled.csv`. They don't match the notebook's current fit,
which uses a different 6-feature set.

`brute_force_fib(coefficients, span, step, indices=...)` grid-searches every combination of
`base ± span` in increments of `step` for the coefficients in `indices` (default: all features,
not the intercept), and returns a DataFrame of every vector sorted by correlation. It reimplements
the scoring with numpy (`_score_coefficient_batch`) rather than calling `evaluate_fib` per vector,
so **any change to `normalize_fib` or `take_paired_seasons` must be mirrored there**; checking
that `brute_force_fib(span=0, ...)` reproduces `evaluate_fib`'s correlation catches drift. It
raises above `max_combinations` (default 1,000,000). Two invariances make parts of any grid
redundant: the intercept never changes the correlation (per-year z-scoring cancels it), nor does
multiplying all feature coefficients by one positive factor.

`load_bowler_stats` raises `KeyError` if the CSV has no `next_year_economy` column, so a CSV
predating this split fails immediately with a pointer to re-run the notebook rather than blowing up
later.

### Normalization

`raw_FIB` is on an arbitrary scale, so `normalize_fib` rescales it per year to the same mean and
standard deviation as that year's `Economy`, producing `FIB` in economy-rate units. Per-year
normalization is deliberate — it absorbs league-wide scoring inflation, so `FIB` is comparable to
`Economy` within a season. A consequence worth knowing: `FIB`'s per-year mean and std always equal
`Economy`'s exactly, so `verbose=True` matching is a tautology, not evidence the model is good.

## Domain rules encoded in the aggregation

These cricket-specific decisions are embedded in cell 11 and are easy to break:

- **Legal balls**: total rows minus deliveries whose `ExtraType` is in
  `wides, noballs, penalty,wides, byes,noballs, legbyes,noballs, penalty`. `Overs = legal balls / 6`.
- **Runs charged to the bowler**: `BatsmanRun` + wide runs + no-ball runs. Byes and leg-byes are
  *not* charged. For `penalty,wides` the fixed 5-run penalty is subtracted (`max(0, sum - 5)`), and
  `byes,noballs` / `legbyes,noballs` are charged only 1 run each (the no-ball itself).
- **`TotalWickets`** (all bowler-credited dismissals): `bowled, caught, lbw, stumped, hit wicket,
  caught and bowled`. Run-outs and retirements are excluded since they aren't credited to a bowler.
- **Fielding-independent dismissals** are split into separate feature columns (`BowledWickets`,
  `LbwWickets`, `HitWicketWickets`, `CaughtAndBowledWickets`, `StumpedWickets`) so the regression can
  weight each on its own. `caught` is intentionally excluded from the features — it depends on a
  fielder, which is the whole premise of the statistic. `stumped` is the judgment call the author
  flagged as uncertain (see the notebook's markdown notes).
- **Known data gap**: overthrows are not represented in the dataset and are currently ignored. The
  noted future fix is scraping commentary data.

## Data files

| File | Role | Key |
| --- | --- | --- |
| `ball-data.csv` | Input, ~295k deliveries | `ID, Innings, Overs, BallNumber` |
| `match-info.csv` | Input, ~1.2k matches | `match_number` → renamed `ID` |
| `bowler-stats.csv` | Generated by cell 17, 0–1 scaled features, all years from 2008, 1072 rows | `Year, Bowler` |
| `bowler-stats-unscaled.csv` | Generated by cell 17, same rows with raw counts | `Year, Bowler` |
| `bowler-stats-with-fib.csv` | **Stale artifact** — see below | `Year, Bowler` |

`bowler-stats-unscaled.csv` doesn't exist until the notebook is re-run after the cell 15–17 change
that introduced it; the committed `bowler-stats.csv` predates the min-max fix too (its
`Economy_avg`/`Economy_diff` aren't 0–1). The fix doesn't change the 8 `FEATURE_COLUMNS`, whose
minimum is 0.

`bowler-stats-with-fib.csv` is no longer written by any code path. It is a frozen snapshot of a
past experiment, and its numbers do **not** correspond to `DEFAULT_COEFFICIENTS`: it was generated
with `StumpedWickets = -50.2354` rather than the current `-11.2354` (verified — every other
coefficient matches, and the residual of that fit is ~1e-15). Don't treat it as ground truth for
regression-testing changes to `fib.py`.

`Bowler` names are the dataset's own initials-style strings (e.g. `Z Khan`, `M Muralitharan`) and
the author notes the naming scheme is inconsistent, though unique per player. Join on names only
within this dataset.

## Conventions

- The notebook's aggregation and next-season pairing (cells 11–12) use explicit Python loops
  (`iterrows`, nested `for`) rather
  than vectorized pandas. Dataset size makes this tolerable; match the surrounding style rather
  than rewriting a cell to be vectorized unless asked. `fib.py` is vectorized throughout — keep it
  that way, since it's the part that gets re-run repeatedly during tuning.
- `fib.py` functions take and return DataFrames without mutating their argument (each does
  `df.copy()` before adding a column), so they can be chained or called out of order safely.
- Cell comments are terse one-liners above each block, and markdown cells carry open questions and
  "MAYBE" notes about feature choices. Keep that record intact when editing — it's the project's
  design log (cell 25 is the running "What I did / Next steps" log). Cell 20 still carries a
  commented-out block of 5 stale coefficients from an older feature set; it doesn't match the
  current model.
