# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Builds "FIB" (Fielding Independent Bowling), a cricket statistic intended to predict a bowler's
*next-season* economy rate better than their *current* economy rate does, by using only outcomes a
bowler controls without help from fielders. Modeled on baseball's FIP. Data is IPL ball-by-ball
data from https://github.com/ritesh-ojha/IPL-DATASET.

Two files do the work. [main.ipynb](main.ipynb) handles data cleaning, per-bowler-season
aggregation, pairing each season with the bowler's next one, and fitting the regression — it ends
once the coefficients are printed.
[fib.py](fib.py) takes a coefficient vector and scores it. There is no package, no test suite, and
no lint config.

## Environment

`.venv/` is a plain `python -m venv` (Python 3.13.7, Homebrew) and is git-ignored via `.venv/.gitignore`.
There is **no requirements.txt or pyproject.toml** — dependencies were pip-installed ad hoc. Currently:
pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, matplotlib 3.11.1, seaborn 0.13.2, ipykernel/jupyter.
If you add an import, install it into `.venv` and mention it to the user; consider proposing a
requirements file rather than silently creating one.

```bash
source .venv/bin/activate
jupyter lab main.ipynb          # or run cells via the VS Code kernel ".venv (3.13.7)"
python fib.py                   # score the default coefficient vector
```

The notebook's saved kernel is `.venv (3.13.7)`; select that kernel, not a global Python.

## Pipeline

Stages 1–4 are the notebook, which is a linear chain of in-memory DataFrames and must be run top to
bottom. Stage 5 is `fib.py`, which is independent — it reads `data/bowler-stats.csv` from disk, so
it only needs the notebook to have been run at some earlier point.

1. **Load & clean** (cells 2–10) → `filtered_df`
   Merges `data/ball-data.csv` with `data/match-info.csv` on `ID` (renamed from `match_number`),
   drops identity/result columns, sorts by `match_date, Innings, Overs, BallNumber`, adds `Year`
   derived from `match_date`.
2. **Aggregate per bowler-season** (cell 11) → `bowler_df`
   Nested loop over `Year` × `Bowler`. **Bowler-seasons under 100 legal balls are skipped.**
3. **Pair with the following season** (cell 12) → `next_year_economy` column, and the whole frame
   is written to `data/bowler-stats.csv` (this cell is the only write).
   Loop looking up the same `Bowler` at `Year + 1`; blank where that bowler has no qualifying season
   the year after (467 of 1072 rows). This is the target `fib.py` scores against, and it lives here
   because it's a property of the data, not of the coefficient vector being tuned.
4. **Fit** (cells 14–16) → `fib_reg`
   Fits a `LinearRegression` on 2015+ seasons, regressing per-over feature rates against `Economy`.
   Cell 15 plots predicted vs actual; cell 16 prints the coefficients. **The notebook ends here** —
   copy the printed coefficients into `fib.py` to score them.
5. **Score** — `fib.py`, run standalone or imported.

### Scoring a coefficient vector (`fib.py`)

`evaluate_fib(coefficients)` applies a vector, normalizes, drops seasons with no following season
(`take_paired_seasons`, which just filters on the `next_year_economy` column the notebook wrote —
it does **not** compute the pairing), and prints `corr(FIB, next_year_economy)` against
`corr(Economy, next_year_economy)`. **That comparison is the project's success metric**: FIB is only
an improvement if it beats plain economy rate. The intended loop is to try a vector, read the
correlation, and repeat:

```python
from fib import evaluate_fib
evaluate_fib([6.1821, 5.7628, 1.9969, 2.8617, -2.5032, -4.5166, 1.4290, -1.4063, -11.2354])
```

`plot=True` draws the two-panel comparison scatter; `verbose=True` prints per-year normalization
tables. The function **never writes to disk**, so rapid tweaking can't dirty a tracked file.

Element 0 of the vector is the intercept; elements 1–8 align positionally with `FEATURE_COLUMNS`.
`compute_raw_fib` validates the length and raises, so a mismatched vector fails loudly rather than
silently mis-mapping — but **reordering `FEATURE_COLUMNS` still silently changes what a given vector
means**, since length is unchanged. As of this writing `DEFAULT_COEFFICIENTS` is exactly the fitted
output of cell 16, not a hand-tuned divergence from it; it's the starting point for tuning.

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

All four CSVs in [data/](data/) are committed:

| File | Role | Key |
| --- | --- | --- |
| `ball-data.csv` | Input, ~295k deliveries | `ID, Innings, Overs, BallNumber` |
| `match-info.csv` | Input, ~1.2k matches | `match_number` → renamed `ID` |
| `bowler-stats.csv` | Generated by cell 12, all years from 2009, 1072 rows | `Year, Bowler` |
| `bowler-stats-with-fib.csv` | **Stale artifact** — see below | `Year, Bowler` |

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
  design log. Cell 15 still carries a commented-out block of 5 stale coefficients from an older
  feature set; it doesn't match the current 8-feature model.
