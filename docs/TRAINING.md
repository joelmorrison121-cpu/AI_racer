# Training notes

## What happens in one generation

All 30 agents start at the same spot and advance one fixed 1/60 s step at a time. Cars pass through each other, so every driver faces the same track. An agent stops when it crashes, makes no progress for 8 s, reaches the 30 s limit or completes a lap. When every agent has stopped, the generation is ranked by fitness. The best network is saved if it beats the run's record, `history.json` is updated and the next population is bred.

## Breeding (`evolution.py`)

1. Copy the 2 best networks unchanged (elitism).
2. Fill the next 24 places with mutated copies of randomly chosen parents from the top 6. Each weight and bias has a 15% chance of receiving Gaussian noise (σ = 0.2).
3. Add 4 brand-new random networks.

There is no crossover, and no weight changes during an attempt.

## Live viewer details

Colours mark population slots, not families. A gold ring marks the strongest *active* driver; once every attempt has ended, it marks the generation winner. Faded cars have ended unsuccessfully, and cars that completed a lap stay solid. The playback speed keys run more fixed steps per frame and never use a larger timestep, so physics stays identical at any speed.

## Fitness modes

- **Default:** `checkpoints passed × 1000` plus progress along the centre line towards the next checkpoint (0–999). A completed lap adds `LAP_TIME_BONUS × (1 − lap time ÷ time limit)`.
- **`--legacy-fitness` (v1.0):** partial credit uses straight-line distance to the next checkpoint's centre, and every finisher scores exactly `checkpoints × 1000`.

Progress is found by projecting the car onto the nearest segment of the centre line. After the first step, each car only searches about ±150 px of the line around its previous position. On a hairpin, the other leg can be closer as the crow flies, but it is never mistaken for the car's actual position.

## Known limitations

- Each generation is evaluated on one track, so drivers can overfit its corner types. The experiment in `docs/results/` shows this clearly on the unseen technical track.
- Inputs only point to the next checkpoint, so the network can't see a hairpin coming beyond it.
- Breeding uses mutation only (no crossover), with fixed mutation settings.
- Resuming continues from the last *completed* generation; a generation interrupted midway restarts.
