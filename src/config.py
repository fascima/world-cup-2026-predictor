"""Configuration for the Elo-only World Cup predictor.

All subjective modeling assumptions should live here so they can be changed
without digging through the modeling code.
"""

# Starting rating assigned to teams before their first match.
INITIAL_ELO = 1000

# Earliest match year included when building ratings.
MIN_MATCH_YEAR = 2010

# First year included in model backtesting. Earlier matches are still used to
# warm up ratings before scoring begins.
BACKTEST_START_YEAR = 2018

# Whether older matches should have lower impact on Elo updates.
USE_TIME_DECAY = True

# Number of years after which a match has half its original K-factor weight.
TIME_DECAY_HALF_LIFE_YEARS = 16

# Controls how strongly Elo differences map to expected score.
ELO_SCALE = 300

# Elo points added to the home team for non-neutral matches.
HOME_ADVANTAGE_ELO = 50

# K-factors by competition class; higher values move ratings more per match.
K_FACTORS = {
    "friendly": 20,
    "qualifier": 30,
    "continental": 40,
    "world_cup": 60,
    "default": 30,
}

# Global multiplier applied to every Elo update after K-factor and time decay.
ELO_UPDATE_MULTIPLIER = 0.75

# Whether match updates should be exactly zero-sum. Backtests may decide
# whether this should be enabled for the default model.
USE_ZERO_SUM_ELO_UPDATES = False

# Whether larger wins should move Elo ratings more than narrow wins.
USE_MARGIN_OF_VICTORY = True

# Margin multiplier method. "one_plus_log" makes a 1-goal win count as 1.0.
MARGIN_OF_VICTORY_METHOD = "one_plus_log"

# Whether expected multi-goal wins by big favorites should receive less extra credit.
USE_FAVORITE_MISMATCH_DAMPENER = True

# Elo gap where favorite-win margin dampening starts.
FAVORITE_MISMATCH_START_ELO_GAP = 100

# Controls how quickly the dampener shrinks as favorite Elo gap grows.
FAVORITE_MISMATCH_SCALE = 400

# Lower bound for favorite-win margin dampening.
MIN_FAVORITE_MISMATCH_DAMPENER = 0.65

# Whether Elo updates should be scaled by the opponent's absolute rating.
USE_OPPONENT_STRENGTH_MULTIPLIER = True

# Opponent Elo that receives a neutral multiplier of about 1.0.
OPPONENT_STRENGTH_BASELINE_ELO = 1000

# Controls how quickly the opponent-strength multiplier changes with Elo.
OPPONENT_STRENGTH_SCALE = 400

# Lower bound so matches against weaker teams still have some influence.
MIN_OPPONENT_STRENGTH_MULTIPLIER = 0.60

# Upper bound so matches against elite teams do not become too extreme.
MAX_OPPONENT_STRENGTH_MULTIPLIER = 1.40

# Whether predictions should adjust team ratings by recent opponent quality.
USE_SCHEDULE_STRENGTH_ADJUSTMENT = True

# Number of prior opponents used for recent schedule strength.
SCHEDULE_STRENGTH_WINDOW = 20

# Baseline opponent Elo that creates no schedule adjustment.
SCHEDULE_STRENGTH_BASELINE_ELO = 1000

# Fraction of schedule strength above/below baseline added to prediction rating.
SCHEDULE_STRENGTH_WEIGHT = 0.15

# Cap for schedule-strength adjustment in Elo points.
MAX_SCHEDULE_STRENGTH_ADJUSTMENT = 75

# Whether predictions should blend long-term Elo with a faster recent-form Elo.
USE_RECENT_FORM_RATING = True

# Weight of recent-form Elo in the prediction rating.
RECENT_FORM_WEIGHT = 0.20

# K-factor multiplier used for the recent-form Elo update.
RECENT_FORM_K_MULTIPLIER = 2.0

# Whether to replay the training history multiple times before final scoring.
USE_ITERATIVE_ELO = True

# Number of warm-up passes used when iterative Elo is enabled.
ITERATIVE_ELO_PASSES = 2

# Baseline draw probability used to convert Elo expected score into W/D/L odds.
BASE_DRAW_PROB = 0.28

# Lower bound for draw probability, even with a large Elo gap.
MIN_DRAW_PROB = 0.12

# Controls how quickly draw probability falls as Elo difference grows.
DRAW_ELO_SCALE = 2000

# Whether to learn draw probability from historical Elo-gap buckets.
USE_EMPIRICAL_DRAW_PROB = False

# Absolute adjusted Elo-gap bucket edges for empirical draw rates.
DRAW_ELO_BINS = [0, 50, 100, 150, 200, 300, 450, 1000000]

# Number of prior pseudo-matches used to smooth each draw bucket.
EMPIRICAL_DRAW_PRIOR_MATCHES = 200

# Reserved for future versions; this Elo-only version uses no temperature.
USE_TEMPERATURE = False

# Seed for Monte Carlo simulations. None means each run uses fresh randomness.
MONTE_CARLO_SEED = None

# Seed for the saved sample bracket. None means each run uses fresh randomness.
SAMPLE_BRACKET_SEED = None

# Whether World Cup simulations should adjust ratings by current squad market value.
USE_MARKET_VALUE_ADJUSTMENT = True

# Log-scale Elo points per natural-log increase in market value ratio.
MARKET_VALUE_ELO_SCALE = 35

# Maximum absolute Elo adjustment from market value.
MAX_MARKET_VALUE_ELO_ADJUSTMENT = 125

# Minimum positive market value accepted when parsing data.
MIN_MARKET_VALUE_EUR = 1_000_000

# Whether the separate Poisson goal backtest should learn average total goals
# from its training split instead of using POISSON_DEFAULT_TOTAL_GOALS.
POISSON_USE_TRAINING_AVG_TOTAL_GOALS = True

# Fallback average total goals per match for the separate Poisson goal model.
POISSON_DEFAULT_TOTAL_GOALS = 2.6

# Lower bound for one team's expected goals in the Poisson goal model.
POISSON_MIN_EXPECTED_GOALS = 0.15

# Upper bound for one team's expected goals in the Poisson goal model.
POISSON_MAX_EXPECTED_GOALS = 4.5

# Highest scoreline included when summing Poisson outcome probabilities.
POISSON_MAX_GOALS = 10

# Multiplier applied to draw scoreline mass in the separate Poisson model.
# A value above 1.0 is a simple draw-inflated Poisson correction.
POISSON_DRAW_INFLATION = 1.2

# Whether the separate Poisson model should lower total expected goals for
# close Elo matchups and slightly raise them for large mismatches.
POISSON_USE_ELO_GAP_TOTAL_GOALS = True

# Elo gap below which the strongest low-scoring close-match adjustment applies.
POISSON_TOTAL_GOALS_CLOSE_GAP = 50

# Elo gap below which the medium low-scoring close-match adjustment applies.
POISSON_TOTAL_GOALS_MEDIUM_GAP = 100

# Elo gap below which the small low-scoring close-match adjustment applies.
POISSON_TOTAL_GOALS_SMALL_GAP = 150

# Elo gap below which no total-goals adjustment is applied.
POISSON_TOTAL_GOALS_NEUTRAL_GAP = 250

# Total-goals adjustment for very close Elo matchups.
POISSON_TOTAL_GOALS_CLOSE_ADJUSTMENT = -0.45

# Total-goals adjustment for moderately close Elo matchups.
POISSON_TOTAL_GOALS_MEDIUM_ADJUSTMENT = -0.30

# Total-goals adjustment for slightly close Elo matchups.
POISSON_TOTAL_GOALS_SMALL_ADJUSTMENT = -0.15

# Total-goals adjustment for large Elo mismatches.
POISSON_TOTAL_GOALS_MISMATCH_ADJUSTMENT = 0.10

# Whether the separate Poisson backtest should classify close, draw-heavy
# probability rows as draws even when draw is not the single largest probability.
POISSON_USE_DRAW_DECISION_RULE = True

# Minimum draw probability required for the Poisson draw decision rule.
POISSON_DRAW_DECISION_THRESHOLD = 0.30

# Maximum gap between home-win and away-win probabilities for the draw decision rule.
POISSON_DRAW_DECISION_MAX_WIN_PROB_GAP = 0.12

# Whether the separate Poisson model should use historical goals for/against
# profiles in addition to Elo expected score.
POISSON_USE_GOAL_PROFILE = True

# Weight given to goal-profile attack/defense multipliers.
POISSON_GOAL_PROFILE_WEIGHT = 0.20

# Prior pseudo-matches used to shrink team goal profiles toward average.
POISSON_GOAL_PROFILE_PRIOR_MATCHES = 12

# Lower bound for attack/defense goal-profile multipliers.
POISSON_MIN_GOAL_PROFILE_MULTIPLIER = 0.60

# Upper bound for attack/defense goal-profile multipliers.
POISSON_MAX_GOAL_PROFILE_MULTIPLIER = 1.60

# Whether the separate Poisson model should apply a Dixon-Coles low-score adjustment.
USE_DIXON_COLES = True

# Dixon-Coles rho parameter. Negative values generally raise low-score draw probability.
DIXON_COLES_RHO = -0.08

# Lower bound for Dixon-Coles rho to keep low-score adjustments stable.
DIXON_COLES_MIN_RHO = -0.30

# Upper bound for Dixon-Coles rho to keep low-score adjustments stable.
DIXON_COLES_MAX_RHO = 0.30

# Whether to enable the optional enhanced Dixon-Coles lambda feature layer.
USE_ENHANCED_DIXON_COLES = True

# Number of prior team matches used for enhanced recent-form and goal features.
ENHANCED_DC_ROLLING_WINDOW = 10

# Prior pseudo-matches used to shrink enhanced rolling stats toward neutral.
ENHANCED_DC_PRIOR_MATCHES = 6

# Log-lambda weight applied to Elo difference after scaling by this Elo amount.
ENHANCED_DC_ELO_DIFF_WEIGHT = 0.02

# Elo points used to scale the enhanced Elo-difference feature.
ENHANCED_DC_ELO_DIFF_SCALE = 300

# Log-lambda weight applied to natural-log squad market value ratio.
ENHANCED_DC_MARKET_VALUE_WEIGHT = 0.02

# Log-lambda weight applied to recent points-per-match difference.
ENHANCED_DC_RECENT_FORM_WEIGHT = 0.03

# Log-lambda weight applied to rolling attack/defense goal-rate advantage.
ENHANCED_DC_GOAL_RATE_WEIGHT = 0.04

# Log-lambda weight applied to rest-day difference.
ENHANCED_DC_REST_DAYS_WEIGHT = 0.005

# Rest days are capped before computing the enhanced rest feature.
ENHANCED_DC_MAX_REST_DAYS = 14

# Maximum absolute log adjustment applied to one team's expected goals.
ENHANCED_DC_MAX_LOG_LAMBDA_ADJUSTMENT = 0.15

# Multipliers for total expected goals by tournament class.
ENHANCED_DC_TOURNAMENT_TOTAL_GOALS_MULTIPLIERS = {
    "friendly": 1.01,
    "qualifier": 0.99,
    "continental": 0.98,
    "world_cup": 0.98,
    "default": 1.00,
}

# Multiplier for total expected goals on neutral-site matches.
ENHANCED_DC_NEUTRAL_TOTAL_GOALS_MULTIPLIER = 0.99

# Multiplier for total expected goals on non-neutral home/away matches.
ENHANCED_DC_HOME_SITE_TOTAL_GOALS_MULTIPLIER = 1.01

# Scoreline heuristic for favorite wins by absolute Elo gap bucket.
FAVORITE_MARGIN_PROBS = {
    "close": {1: 0.70, 2: 0.22, 3: 0.06, 4: 0.02},
    "medium": {1: 0.55, 2: 0.30, 3: 0.11, 4: 0.04},
    "large": {1: 0.40, 2: 0.35, 3: 0.17, 4: 0.08},
}

# Scoreline heuristic for underdog wins.
UNDERDOG_MARGIN_PROBS = {1: 0.80, 2: 0.16, 3: 0.04, 4: 0.00}

# Draw scoreline probabilities.
DRAW_SCORELINE_PROBS = {0: 0.25, 1: 0.45, 2: 0.25, 3: 0.05}

# Loser goals distribution used when generating non-draw scorelines.
LOSER_GOALS_PROBS = {0: 0.55, 1: 0.35, 2: 0.10}
