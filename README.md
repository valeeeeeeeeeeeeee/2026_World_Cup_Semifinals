# 2026 World Cup Semifinals — ML Prediction Model

Predicts the winners of the 2026 FIFA World Cup semifinals:

- **France vs Spain** — Arlington, 2026-07-14
- **Argentina vs England** — Atlanta, 2026-07-15

## Data

`data/results.csv` is the full **49,509-match** international results
history (1872–2026) from [martj42/international_results](https://github.com/martj42/international_results),
including every 2026 World Cup match played so far (group stage through
quarterfinals) and the two upcoming semifinal fixtures. All models are fit
using only matches strictly **before 2026-07-14**, so no semifinal (or
later) result leaks into training.

## Methodology

Three components, combined into a final ensemble:

1. **Elo ratings** (`src/elo.py`) — every match in history is replayed
   chronologically to build a running Elo rating per team, using the
   standard [World Football Elo](https://www.eloratings.net/about) rules:
   K-factor scaled by competition importance (World Cup > continental
   championships > qualifiers/Nations League > friendlies) and by goal
   difference, plus a home-advantage bonus for non-neutral matches. Elo
   ratings are the core team-strength signal fed into everything else.

2. **Win/Draw/Loss classifier** (`src/features.py`, `src/train_model.py`)
   — a logistic regression trained on `elo_diff`, `neutral`, and rolling
   10-match form (points-per-game and goal-difference deltas), all
   computed leak-free (pre-match state only). Evaluated on a time-based
   holdout (train < 2019, validate ≥ 2019): **60.7% accuracy** / **0.863
   log-loss**, vs. a 1.05 log-loss naive baseline that just guesses the
   class priors.

3. **Poisson goal model** (`src/poisson_model.py`) — Dixon-Coles-style
   attack/defense strengths per team, fit via L2-regularized Poisson
   regression (`sklearn.PoissonRegressor`) on matches since 2015 (to
   reflect current squads). Regularization strength was chosen via a
   2015–2023 train / 2024–2026 validation split minimizing mean Poisson
   deviance.

Because a semifinal **cannot end in a draw**, `src/predict_semifinals.py`
resolves draws two ways and averages them:

- **Monte Carlo simulation** (50,000 runs): regulation goals drawn from
  the Poisson model; a level match goes to extra time (goals scaled to
  30 minutes) and then, if still level, a penalty shootout modeled as a
  mildly Elo-tilted coin flip (shootouts are close to random, but not
  perfectly so).
- **Classifier draw redistribution**: the classifier's 90-minute draw
  probability mass is reassigned to each side using the same shootout
  tilt.

The final reported probability is the average of these two independent
estimates.

## Results

| Match | Elo | Poisson xG | Final advance probability | Prediction |
|---|---|---|---|---|
| France vs Spain | 2243 vs 2266 | 1.07 – 1.38 | **France 45.6% – Spain 54.4%** | **Spain** |
| Argentina vs England | 2264 vs 2179 | 1.18 – 0.92 | **Argentina 61.3% – England 38.7%** | **Argentina** |

**Predicted final: Argentina vs Spain.**

Full breakdown (Elo, expected goals, Monte Carlo probs, classifier probs)
is in `results/semifinal_predictions.json`.

## Running it

```bash
pip install -r requirements.txt
cd src
python train_model.py        # trains + saves the W/D/L classifier
python predict_semifinals.py # runs the full pipeline, prints + saves predictions
```

## Project layout

```
data/                      results.csv, shootouts.csv, former_names.csv (source data)
src/elo.py                 Elo rating engine
src/features.py            leak-free chronological feature builder
src/train_model.py         trains + selects the W/D/L classifier
src/poisson_model.py       Dixon-Coles style Poisson goal model
src/predict_semifinals.py  full pipeline -> final predictions
results/                   trained model + metrics + predictions (generated)
```

## Caveats

- International football outcomes are inherently noisy; a 60% test
  accuracy / sub-1.0 log-loss on a 3-class problem is a solid edge over
  the naive baseline, not a crystal ball.
- Head-to-head history and player-level factors (injuries, suspensions,
  travel/rest) are not modeled — only team-level Elo, recent form and
  scoring rates.
- The penalty-shootout tilt is a modeling assumption (dampened Elo
  logistic), not fit to shootout-specific historical data.
