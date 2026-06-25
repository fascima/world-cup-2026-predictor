# World Cup 2026 Predictor

This project is a 2026 FIFA World Cup predictor for international soccer. It loads historical match results, builds national-team Elo ratings, includes separate Poisson and Dixon-Coles experiments, trains a first supervised logistic regression baseline, and can simulate the World Cup when complete group fixtures are available.

## What It Does

- Loads historical international football results from `data/raw/results.csv`
- Cleans dates, scores, neutral-site flags, and match outcomes
- Builds chronological Elo ratings for national teams
- Backtests match probabilities from 2018 onward with log loss, Brier score, and accuracy
- Saves current ratings to `results/current_elo_ratings.csv`
- Saves historical pre-match Elo predictions to `results/elo_match_predictions.csv`
- Optionally adjusts World Cup simulation ratings with current squad market values from `data/fixtures/team_market_values.csv`
- Saves World Cup simulation ratings to `results/world_cup_team_model_ratings.csv`
- Simulates the 2026 World Cup using the Elo model when `data/fixtures/world_cup_2026_groups.csv` has 12 complete groups
- Saves stage probabilities to `results/team_stage_probabilities.csv`
- Saves one sample tournament bracket to `results/sample_bracket.json`
- Builds a supervised ML feature dataset at `data/processed/ml_match_features.csv`
- Trains a multinomial logistic regression outcome classifier at `models/logistic_match_outcome.joblib`
- Saves logistic regression metrics, confusion matrix, and coefficients under `results/`

## How Elo Works

Every team starts with the same rating. Before each match, the model compares the two teams' ratings. A team with a higher rating has a higher expected score. After the match, ratings move based on the difference between the actual result and the expected result.

In this project:

- A win counts as `1`
- A draw counts as `0.5`
- A loss counts as `0`
- Non-neutral home teams receive a configurable home-advantage boost
- More important competitions can use larger K-factors
- Larger wins can move ratings more when margin of victory is enabled
- Older matches can be down-weighted with time decay

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Add historical results at:

```text
data/raw/results.csv
```

Then run:

```bash
python3 main.py
```

By default, `main.py` also trains the logistic regression baseline. Set `TRAIN_LOGISTIC_MODEL = False` near the top of `main.py` to skip that section.

If `data/fixtures/world_cup_2026_groups.csv` contains complete groups `A` through `L`, with 4 non-placeholder teams in each group, the script also runs 1,000 World Cup simulations.

Run the backtest:

```bash
python3 backtest.py
```

The backtest scores predictions from `BACKTEST_START_YEAR` onward after using earlier matches to warm up ratings.

Refresh live World Cup fixtures, completed results, current Elo ratings, and today's dashboard predictions:

```bash
export FOOTBALL_DATA_API_KEY=your_api_key_here
python3 scripts/update_world_cup_live_data.py
```

The updater uses football-data.org's World Cup match endpoint, caches match data under `data/live/`, writes a historical-plus-live results file, updates current Elo/model ratings from completed matches, and writes `results/todays_match_predictions.csv` for the dashboard.

Run the historical World Cup market-value backtest:

```bash
python3 market_value_backtest.py
```

This compares Elo-only predictions against Elo plus historical tournament-year market values.

## Logistic Regression ML Model

The logistic regression model is the project's first supervised machine learning model. It predicts a three-class match outcome:

- `0 = team_a_win`
- `1 = draw`
- `2 = team_b_win`

It uses only pre-match features: pre-match Elo values, prediction Elo values, adjusted Elo gap, close-match flags, Elo-derived probabilities, recent form, recent goals for and against, recent points, tournament type, and neutral-site status. Rolling team features are calculated from matches before the current match date and are updated only after that date's matches are processed, which prevents the current result from leaking into its own features.

Training uses a chronological split instead of a random split:

- Training: matches before `2022-01-01`
- Test: matches from `2022-01-01` onward

This model is intended as an interpretable baseline before adding XGBoost or more advanced models. It can be compared against Elo, Poisson, and Dixon-Coles outputs, and its saved probabilities use the same `team_a_win_prob`, `draw_prob`, and `team_b_win_prob` naming used by the other prediction helpers.

The logistic training path also tunes regularization strength and draw class weighting on a chronological validation window inside the training period. The 2022+ test set remains untouched until final evaluation.

For simulator symmetry, the training folds can be augmented with flipped team-order rows. In the current configuration, flipped decisive-result rows are labeled as draws to deliberately increase draw pressure; the original saved feature dataset still contains one real row per historical match.

Train it with the normal project entry point:

```bash
python3 main.py
```

Outputs:

- Feature dataset: `data/processed/ml_match_features.csv`
- Trained model: `models/logistic_match_outcome.joblib`
- Metrics: `results/logistic_model_metrics.csv`
- Confusion matrix: `results/logistic_confusion_matrix.csv`
- Coefficients: `results/logistic_coefficients.csv`
- Tuning results: `results/logistic_tuning_results.csv`

Run the separate Poisson goal-model backtest:

```bash
python3 poisson_backtest.py
```

This leaves the main Elo model unchanged. It uses pre-match Elo expected score as an input, converts it into expected goals, and derives win/draw/loss probabilities by summing Poisson scoreline probabilities.

Run the World Cup-only Poisson market-value backtest:

```bash
python3 poisson_market_value_backtest.py
```

This compares the separate Poisson model against Poisson plus historical tournament-year market values.

Tune the separate Poisson model:

```bash
python3 poisson_tuning.py
```

This grid-searches Dixon-Coles rho, draw inflation, goal-profile weight, and draw decision-rule thresholds, then writes `results/poisson_tuning_results.csv`.

Run rolling-window validation for the separate Poisson model:

```bash
python3 poisson_rolling_validation.py
```

This tunes the same Poisson settings across multiple validation windows instead of one combined holdout.

Compare Basic Poisson, Dixon-Coles, and Enhanced Dixon-Coles:

```bash
python3 enhanced_dixon_coles_backtest.py
```

This keeps the existing Poisson and Dixon-Coles models unchanged, then adds a third model that adjusts expected-goals lambdas before applying Dixon-Coles. It writes `results/enhanced_dixon_coles_predictions.csv` and `results/enhanced_dixon_coles_summary.csv`.

## StatsBomb Open Data Parser

If StatsBomb Open Data has been downloaded under `data/external/statsbomb_open_data`, convert the World Cup event files into team-match features with:

```bash
python3 statsbomb_parser.py
```

The parser writes one row per team per match to:

```text
data/processed/statsbomb_team_match_features.csv
```

The output includes match metadata, result fields, xG, non-penalty xG, shot quality, pass/carry progression, final-third and box entries, pressures, counterpressures, defensive actions, and mirrored opponent columns such as `opponent_xg` and `opponent_shots`. Penalty shootout events are excluded from the aggregate features.

When this parsed file exists, `main.py` also builds rolling pre-match StatsBomb priors at:

```text
data/processed/statsbomb_team_rolling_features.csv
```

Those priors use only matches dated before the prediction date and are merged into the supervised ML feature table as a small set of xG, shot, box-entry, and pressure signals plus missingness flags.

## Expected Data Format

`data/raw/results.csv` should contain columns similar to:

```text
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
```

Required columns:

- `date`
- `home_team`
- `away_team`
- `home_score`
- `away_score`

Optional but used when present:

- `tournament`
- `neutral`

The `neutral` column can contain booleans or strings such as `TRUE`, `FALSE`, `True`, or `False`.

The World Cup group file should use:

```text
group,team
A,Mexico
A,Canada
```

Placeholder teams such as `Team TBD`, `TBD`, `Playoff Winner`, and blank names are ignored.

Optional World Cup squad market values can be added at:

```text
data/fixtures/team_market_values.csv
```

Use this format:

```text
team,market_value_eur
France,1200000000
England,1100000000
Brazil,950m
```

Values may be plain euro amounts or readable values such as `950m` or `1.2bn`. Team names must match the names in `world_cup_2026_groups.csv`.

Historical World Cup market values for backtesting should be added at:

```text
data/fixtures/historical_world_cup_market_values.csv
```

Use this format:

```text
tournament_year,team,market_value_eur
2006,Brazil,365000000
2010,Spain,650000000
2014,Germany,458870928
2018,France,1035231850
2022,Argentina,633200000
```

The historical market-value backtest scores only actual `FIFA World Cup` tournament matches, not qualifiers. It trains ratings only on matches before each World Cup starts, then scores that tournament chronologically. Team names must match `data/raw/results.csv`.

## Modeling Assumptions

Subjective model choices are stored in `src/config.py` so they are easy to change.

- `INITIAL_ELO = 1000`: starting rating for every team before its first match.
- `MIN_MATCH_YEAR = 2010`: only matches from 2010 onward are used.
- `BACKTEST_START_YEAR = 2018`: first year scored by the backtest; earlier matches still warm up ratings.
- `USE_TIME_DECAY = True`: older matches have less impact on rating updates.
- `TIME_DECAY_HALF_LIFE_YEARS = 16`: a match sixteen years older than the latest match receives half the K-factor weight. An eight-year-old match still counts meaningfully at about 71% weight.
- `ELO_SCALE = 300`: controls how Elo differences translate into expected score.
- `HOME_ADVANTAGE_ELO = 50`: non-neutral home teams receive a 50-point Elo boost for prediction and update expectation.
- `K_FACTORS`: controls how much ratings move by competition type: friendlies, qualifiers, continental tournaments, World Cups, and defaults.
- `ELO_UPDATE_MULTIPLIER = 0.75`: lowers every per-match Elo update without using a hard cap.
- `USE_ZERO_SUM_ELO_UPDATES = False`: zero-sum updates are implemented but disabled by default because the 2018+ backtest was worse with them enabled.
- `USE_MARGIN_OF_VICTORY = True`: bigger wins move ratings more than narrow wins.
- `MARGIN_OF_VICTORY_METHOD = "one_plus_log"`: margin multiplier is `1 + log(goal_margin)`, so a 1-goal win is `1.0`.
- `USE_FAVORITE_MISMATCH_DAMPENER = True`: reduces extra margin-of-victory credit when a clear favorite wins by multiple goals.
- `FAVORITE_MISMATCH_START_ELO_GAP = 100`: favorite-win dampening begins above this adjusted Elo gap.
- `FAVORITE_MISMATCH_SCALE = 400`: controls how quickly favorite-win margin dampening increases as the Elo gap grows.
- `MIN_FAVORITE_MISMATCH_DAMPENER = 0.65`: lower bound for the favorite-win margin dampener.
- `USE_OPPONENT_STRENGTH_MULTIPLIER = True`: scales Elo updates by the opponent's absolute rating.
- `OPPONENT_STRENGTH_BASELINE_ELO = 1000`: opponent rating that receives a neutral multiplier around `1.0`.
- `OPPONENT_STRENGTH_SCALE = 400`: controls how quickly the multiplier rises or falls as opponent Elo changes.
- `MIN_OPPONENT_STRENGTH_MULTIPLIER = 0.60`: lower bound for matches against weaker opponents.
- `MAX_OPPONENT_STRENGTH_MULTIPLIER = 1.40`: upper bound for matches against stronger opponents.
- `USE_SCHEDULE_STRENGTH_ADJUSTMENT = True`: adjusts prediction ratings based on recent opponent quality.
- `SCHEDULE_STRENGTH_WINDOW = 20`: number of prior opponents used for schedule strength.
- `SCHEDULE_STRENGTH_BASELINE_ELO = 1000`: opponent average that creates no schedule adjustment.
- `SCHEDULE_STRENGTH_WEIGHT = 0.15`: fraction of schedule strength added to prediction rating.
- `MAX_SCHEDULE_STRENGTH_ADJUSTMENT = 75`: cap for schedule adjustment.
- `USE_RECENT_FORM_RATING = True`: blends long-term Elo with a faster-moving recent-form Elo.
- `RECENT_FORM_WEIGHT = 0.20`: recent-form share of the prediction rating.
- `RECENT_FORM_K_MULTIPLIER = 2.0`: K-factor multiplier for recent-form Elo updates.
- `USE_ITERATIVE_ELO = True`: replays the training history before final scoring so ratings are less dependent on everyone starting at `INITIAL_ELO`.
- `ITERATIVE_ELO_PASSES = 2`: two passes was the best tested iterative setting on the 2018+ holdout.
- `BASE_DRAW_PROB = 0.28`: baseline draw probability used when turning Elo expected score into win/draw/loss probabilities.
- `MIN_DRAW_PROB = 0.12`: minimum draw probability even when teams are far apart in Elo.
- `DRAW_ELO_SCALE = 2000`: controls how quickly draw probability falls as the Elo gap grows.
- `USE_EMPIRICAL_DRAW_PROB = False`: empirical draw buckets are implemented but disabled by default because the 2018+ backtest was worse than the heuristic.
- `DRAW_ELO_BINS`: absolute Elo-gap buckets used when empirical draw probabilities are enabled.
- `EMPIRICAL_DRAW_PRIOR_MATCHES = 200`: smoothing strength for empirical draw buckets.
- `USE_TEMPERATURE = False`: no temperature or extra randomness is used in this version.
- `MONTE_CARLO_SEED = None`: controls tournament simulation randomness. `None` gives fresh results each run; set an integer for reproducible runs.
- `SAMPLE_BRACKET_SEED = None`: controls the saved sample bracket randomness. `None` gives a fresh bracket each run; set an integer for reproducible sample brackets.
- `USE_MARKET_VALUE_ADJUSTMENT = True`: allows World Cup simulations to adjust ratings using current squad market value when `data/fixtures/team_market_values.csv` is populated.
- `MARKET_VALUE_ELO_SCALE = 35`: controls how many Elo points are added per natural-log increase in market value ratio.
- `MAX_MARKET_VALUE_ELO_ADJUSTMENT = 125`: caps the absolute market-value adjustment so squad value does not dominate Elo.
- `MIN_MARKET_VALUE_EUR = 1_000_000`: ignores missing, invalid, or tiny market values.
- `POISSON_USE_TRAINING_AVG_TOTAL_GOALS = True`: the separate Poisson model learns average total goals from the training split.
- `POISSON_DEFAULT_TOTAL_GOALS = 2.6`: fallback total goals per match for the Poisson model.
- `POISSON_MIN_EXPECTED_GOALS = 0.15`: lower bound for one team's expected goals in the Poisson model.
- `POISSON_MAX_EXPECTED_GOALS = 4.5`: upper bound for one team's expected goals in the Poisson model.
- `POISSON_MAX_GOALS = 10`: highest scoreline included when summing Poisson probabilities.
- `POISSON_DRAW_INFLATION = 1.2`: legacy draw-scoreline multiplier used only when Dixon-Coles is disabled.
- `POISSON_USE_ELO_GAP_TOTAL_GOALS = True`: lowers expected total goals in close Elo matchups and slightly raises them in large mismatches.
- `POISSON_TOTAL_GOALS_CLOSE_GAP = 50`: Elo-gap cutoff for the strongest close-match total-goals adjustment.
- `POISSON_TOTAL_GOALS_MEDIUM_GAP = 100`: Elo-gap cutoff for the medium close-match total-goals adjustment.
- `POISSON_TOTAL_GOALS_SMALL_GAP = 150`: Elo-gap cutoff for the small close-match total-goals adjustment.
- `POISSON_TOTAL_GOALS_NEUTRAL_GAP = 250`: Elo-gap cutoff above which the mismatch total-goals adjustment applies.
- `POISSON_TOTAL_GOALS_CLOSE_ADJUSTMENT = -0.45`: total-goals change for very close Elo matchups.
- `POISSON_TOTAL_GOALS_MEDIUM_ADJUSTMENT = -0.30`: total-goals change for moderately close Elo matchups.
- `POISSON_TOTAL_GOALS_SMALL_ADJUSTMENT = -0.15`: total-goals change for slightly close Elo matchups.
- `POISSON_TOTAL_GOALS_MISMATCH_ADJUSTMENT = 0.10`: total-goals change for large Elo mismatches.
- `POISSON_USE_DRAW_DECISION_RULE = True`: lets the separate Poisson backtest classify close, draw-heavy rows as draws.
- `POISSON_DRAW_DECISION_THRESHOLD = 0.30`: minimum draw probability needed for the Poisson draw decision rule.
- `POISSON_DRAW_DECISION_MAX_WIN_PROB_GAP = 0.12`: maximum home-win versus away-win probability gap allowed for a draw decision.
- `POISSON_USE_GOAL_PROFILE = True`: uses prior goals for and against to adjust expected goals in the separate Poisson model.
- `POISSON_GOAL_PROFILE_WEIGHT = 0.20`: weight given to goal-profile attack/defense multipliers.
- `POISSON_GOAL_PROFILE_PRIOR_MATCHES = 12`: prior pseudo-matches used to shrink goal profiles toward average.
- `POISSON_MIN_GOAL_PROFILE_MULTIPLIER = 0.60`: lower bound for goal-profile multipliers.
- `POISSON_MAX_GOAL_PROFILE_MULTIPLIER = 1.60`: upper bound for goal-profile multipliers.
- `USE_DIXON_COLES = True`: applies a Dixon-Coles low-score adjustment to the Poisson score matrix.
- `DIXON_COLES_RHO = -0.08`: low-score correlation parameter; negative values generally raise low-score draw probability.
- `DIXON_COLES_MIN_RHO = -0.30`: lower bound used to keep Dixon-Coles probabilities stable.
- `DIXON_COLES_MAX_RHO = 0.30`: upper bound used to keep Dixon-Coles probabilities stable.
- `USE_ENHANCED_DIXON_COLES = True`: enables the optional enhanced lambda experiment used by `enhanced_dixon_coles_backtest.py`.
- `ENHANCED_DC_ROLLING_WINDOW = 10`: number of previous team matches used for recent form and rolling goals.
- `ENHANCED_DC_PRIOR_MATCHES = 6`: shrinkage prior so teams with few recent matches do not get extreme feature values.
- `ENHANCED_DC_ELO_DIFF_WEIGHT = 0.02`: small extra log-lambda weight for Elo difference on top of the base Elo split.
- `ENHANCED_DC_MARKET_VALUE_WEIGHT = 0.02`: log-lambda weight for historical squad market-value ratio when available.
- `ENHANCED_DC_RECENT_FORM_WEIGHT = 0.03`: log-lambda weight for recent points-per-match difference.
- `ENHANCED_DC_GOAL_RATE_WEIGHT = 0.04`: log-lambda weight for rolling goals-for and goals-against rates.
- `ENHANCED_DC_REST_DAYS_WEIGHT = 0.005`: log-lambda weight for rest-day difference.
- `ENHANCED_DC_MAX_REST_DAYS = 14`: cap used before comparing rest days.
- `ENHANCED_DC_MAX_LOG_LAMBDA_ADJUSTMENT = 0.15`: cap that prevents enhanced lambdas from moving too far from the base model.
- `ENHANCED_DC_TOURNAMENT_TOTAL_GOALS_MULTIPLIERS`: total-goal environment multipliers by competition class.
- `ENHANCED_DC_NEUTRAL_TOTAL_GOALS_MULTIPLIER = 0.99`: total-goal multiplier for neutral-site matches.
- `ENHANCED_DC_HOME_SITE_TOTAL_GOALS_MULTIPLIER = 1.01`: total-goal multiplier for non-neutral home/away matches.

The scoreline simulation probabilities are also in `src/config.py`. They control how often wins are by 1, 2, 3, or 4 goals and how often draws are 0-0, 1-1, 2-2, or 3-3.

### Opponent Strength Multiplier

The opponent strength multiplier was added to reduce Elo inflation from repeated wins against weaker teams. Pure standard Elo already rewards upsets through the `actual - expected` term, so this is not part of standard Elo. It is a project-specific adjustment that also considers the opponent's absolute quality: positive rating changes against low-rated opponents move less, and positive rating changes against high-rated opponents move more.

For negative rating changes, the logic is reversed. Losing or underperforming against a strong opponent is punished less than losing or underperforming against a weak opponent.

The multiplier is bounded by `MIN_OPPONENT_STRENGTH_MULTIPLIER` and `MAX_OPPONENT_STRENGTH_MULTIPLIER` so it cannot become extreme. To return to a purer Elo update, set `USE_OPPONENT_STRENGTH_MULTIPLIER = False` in `src/config.py`.

### Favorite Mismatch Dampener

The favorite mismatch dampener reduces margin-of-victory credit when a team that was already a clear Elo favorite wins by multiple goals. This is confederation-neutral: it applies equally to any favorite beating a much weaker opponent. Upset wins and draws are not dampened by this rule.

### Market Value Adjustment

Current squad market value can help the World Cup simulation because it adds a forward-looking talent proxy that match results alone may miss. For example, a historically inconsistent team with elite players may be stronger than its recent Elo record suggests.

This adjustment is used only for tournament simulations, not for the historical Elo backtest. Using current 2026 squad values to judge matches from 2018, 2019, or 2020 would leak future information into the test. A proper historical market-value backtest would need squad values as they existed before each match date.

When enabled and when `data/fixtures/team_market_values.csv` has values, the model uses the median World Cup team market value as the baseline. Each team's adjustment is:

```text
log(team_market_value / median_market_value) * MARKET_VALUE_ELO_SCALE
```

The adjustment is capped by `MAX_MARKET_VALUE_ELO_ADJUSTMENT`. Teams without market values receive no market-value adjustment. The final simulation ratings are saved to `results/world_cup_team_model_ratings.csv`.

### Backtest Results

The 2018+ holdout backtest currently scores 7,952 matches. The selected default stack had the best tested log loss among the implemented changes:

```text
baseline before these changes: log_loss 0.893131
zero-sum updates enabled:      log_loss 0.915484
schedule strength only:        log_loss 0.913266
empirical draw enabled:        log_loss 0.916746
selected recent-form blend:    log_loss 0.891591
iterative Elo, 2 passes:       log_loss 0.892697, accuracy 0.599472
```

The full trail is saved in `results/backtest_summary.csv`. Zero-sum updates and empirical draw probabilities remain available as config switches, but they are disabled by default because they worsened this holdout. Iterative Elo is enabled with two passes because it improved accuracy, although the non-iterative recent-form model still had slightly better log loss.

### Historical Market-Value Backtest

`market_value_backtest.py` compares:

- Elo only
- Elo plus historical World Cup market value

It writes match-level predictions to `results/world_cup_market_value_backtest_predictions.csv` and summary metrics to `results/world_cup_market_value_backtest_summary.csv`.

This backtest intentionally uses historical values by tournament year instead of current squad values. It also disables the default `MIN_MATCH_YEAR` filter for this specific test so older tournaments such as 2006 can be scored if `results.csv` contains them.

### Separate Poisson Goal Model

`poisson_backtest.py` is a separate experiment, not a replacement for `main.py` or `backtest.py`.

The current Elo model directly creates win/draw/loss probabilities from Elo expected score and a draw heuristic. The Poisson model does something different:

1. Use the same chronological pre-match Elo expected score.
2. Learn average total goals from the training split.
3. Learn team goal profiles from prior goals for and against.
4. Split expected goals between the two teams based on the Elo expected score.
5. Adjust each team's expected goals using its attack profile and the opponent's defensive profile.
6. Lower total expected goals for close Elo matchups, because close matches tend to be more draw-prone.
7. Build an independent Poisson score matrix from 0-0 through `POISSON_MAX_GOALS`.
8. Apply a Dixon-Coles low-score adjustment to 0-0, 1-0, 0-1, and 1-1.
9. Renormalize the adjusted score matrix.
10. Convert that score matrix into home-win, draw, and away-win probabilities.
11. Apply a separate draw decision rule for classification when draw probability is high and the two win probabilities are close.

Pure independent Poisson still tends to make one side's win probability larger than draw probability at normal soccer scoring levels. The Dixon-Coles adjustment is a focused correction for low soccer scorelines such as 0-0, 1-0, 0-1, and 1-1. It writes `results/poisson_backtest_predictions.csv` and `results/poisson_backtest_summary.csv`.

`enhanced_dixon_coles_backtest.py` compares three models on the same chronological holdout:

- Basic Poisson: the existing expected-goals calculation with an independent Poisson score matrix.
- Dixon-Coles: the existing expected-goals calculation plus the Dixon-Coles low-score adjustment.
- Enhanced Dixon-Coles: the same Dixon-Coles adjustment, but with lambdas adjusted by pre-match Elo difference, historical market-value log difference when available, recent form, rolling goals for and against, rest days, tournament class, and neutral-site status.

The enhanced features are calculated before each match and updated only after the match is scored, so the backtest does not use future results. Historical market values are used only when `data/fixtures/historical_world_cup_market_values.csv` has values for that tournament year; otherwise that feature is neutral.

`poisson_tuning.py` grid-searches the separate Poisson/Dixon-Coles settings on the same 2018+ holdout and writes `results/poisson_tuning_results.csv`.

`poisson_market_value_backtest.py` scores previous World Cups with the separate Poisson model, then repeats the same matches after adjusting the Elo gap with historical tournament-year market values. It writes `results/poisson_world_cup_market_value_predictions.csv` and `results/poisson_world_cup_market_value_summary.csv`.

`poisson_rolling_validation.py` evaluates the Poisson tuning grid across rolling windows: 2018-2019, 2020-2021, 2022-2023, and 2024-2026. It writes `results/poisson_rolling_validation_results.csv` and `results/poisson_rolling_validation_folds.csv`.

## Current Bracket Limitation

The project is designed to use a real World Cup-style bracket. The Round of 32 is slot-based rather than strongest-vs-weakest, and teams are not reseeded after each knockout round.

Some third-place-team matchups depend on which groups produce advancing third-place teams. The Round of 32 slots in `src/bracket.py` encode the allowed third-place source groups for each third-place slot, and the code includes a `THIRD_PLACE_MAPPING` dictionary where an exact official lookup table can be filled in.

The knockout path uses fixed match-number pairings through the final instead of simply pairing adjacent winners after each round.

When an exact lookup mapping is not present yet, the code uses a deterministic provisional assignment that respects each slot's allowed third-place group letters. This keeps third-place finishers moving into the Round of 32 without strongest-vs-weakest reseeding. It should still be replaced with the official FIFA mapping if a full lookup table is added.

## Current Limitations

- The World Cup simulator still uses Elo ratings by default; the logistic regression probabilities are saved for later simulator integration.
- There is optional squad market-value data for World Cup simulations, but no player availability, injuries, travel adjustment, or betting-market calibration.
- The main model still uses a simple draw heuristic because Elo naturally predicts expected score, not exact win/draw/loss probabilities.
- The Poisson goal model is implemented as a separate backtest path and is not yet used by the World Cup simulator.
- The logistic regression model is an interpretable baseline, not a tuned advanced ML model.
- The full official third-place Round of 32 lookup table is not filled in yet; unmapped combinations use a provisional assignment constrained by each slot's allowed third-place groups.

## Future Improvements

- Add the complete official 2026 third-place mapping.
- Calibrate Elo parameters against historical holdout data.
- Compare and calibrate the separate Poisson goal model before deciding whether to use it in simulation.
- Add historical squad values by match date so market-value features can be backtested without leakage.
- Add player availability, injuries, and travel adjustments.
- Build a website or app after the modeling layer is stable.
