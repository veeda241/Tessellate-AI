---
license: mit
library_name: catboost
tags:
  - catboost
  - tabular-regression
  - formula-1
  - strategy
  - motorsport
---

# Tessellate AI — F1 Counterfactual Pit-Strategy Regressor

A CatBoost regressor that predicts a car's **finishing position** from a mid-race snapshot, used twice on the *same* snapshot to answer one question: **BOX NOW or STAY OUT?**

Lower predicted position is better. Scoring the identical state with `will_pit = 1` and `will_pit = 0` and comparing the two projections is the whole counterfactual — no separate "pit model", no post-hoc adjustment.

**This model was trained on a simulated, physics-based data generator, not on real Formula 1 telemetry.** Read [Training data](#training-data) before drawing any conclusion from the metrics.

## Files in this repo

| File | Size | What it is |
| --- | --- | --- |
| `strategy_catboost.cbm` | ~460 KB | Production model, native CatBoost C++ binary (not ONNX, not Pickle — load with `CatBoostRegressor().load_model()`) |
| `model_metadata.json` | ~1.4 KB | Feature schema, hyper-parameters, GroupKFold metrics, feature importances, library versions, `trained_at` |
| `track_constants.json` | ~2 KB | Serving-time circuit memory (`pit_loss`, `traffic_index`, `total_laps` for 20 circuits). The 2026 engine looks the circuit's pit-lane delta up here on every request, so inference is incomplete without it |
| `example_inference.py` | ~1 KB | Loads the `.cbm` and prints the BOX/STAY-OUT call for one snapshot |
| `Tessellate-AI_Architecture_and_Use_Cases.pdf` | ~690 KB | 16-page system walkthrough: modules, counterfactual contract, request lifecycle, limitations |

## Feature schema (11 columns)

| # | Feature | Type | Importance | Meaning |
| --- | --- | --- | --- | --- |
| 1 | `track_id` | categorical | 0.84 | Circuit name, e.g. `"Monaco Grand Prix"` |
| 2 | `laps_remaining` | numeric | 4.26 | Laps left to the flag |
| 3 | `current_position` | numeric | 45.69 | Place on lap at the decision moment |
| 4 | `tire_age` | numeric | 9.17 | Laps run on the current compound |
| 5 | `tire_compound` | categorical | 4.79 | `SOFT` / `MEDIUM` / `HARD` |
| 6 | `gap_ahead` | numeric | 0.68 | Time (s) to the car directly in front |
| 7 | `gap_behind` | numeric | 1.77 | Time (s) to the car directly behind |
| 8 | `pit_loss_s` | numeric | 1.86 | Circuit pit-lane loss (s), from `track_constants.json` |
| 9 | `rejoin_gap_s` | numeric | 1.07 | `gap_ahead − pit_loss_s` — negative means the stop **rejoins behind** the car it was chasing |
| 10 | `driver_pace_index` | numeric | 7.37 | Relative qualifying/pace rating |
| 11 | `will_pit` | numeric (0/1) | 22.50 | The only value that differs between the two counterfactual calls |

Categoricals (`track_id`, `tire_compound`) are handled natively by CatBoost and are already bound inside the `.cbm`; no encoder or vocab file is shipped alongside it.

## Quickstart

```bash
pip install catboost==1.2.10 pandas huggingface_hub
hf download Veeda241/tessellate-f1-strategy --local-dir model
python model/example_inference.py
```

Expected output (verified against the uploaded `.cbm`, not illustrative):

```
British Grand Prix | lap 34 | P6 on 24-lap SOFT
  projected finish  BOX=10.61  STAY=11.52
  rejoin gap -1.10 s (pit loss 21.5 s)
  call -> BOX NOW (+0.90 positions)
```

```python
import pandas as pd
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model("strategy_catboost.cbm")        # native binary; no onnxruntime

FEATURES = ["track_id", "laps_remaining", "current_position", "tire_age",
            "tire_compound", "gap_ahead", "gap_behind", "pit_loss_s",
            "rejoin_gap_s", "driver_pace_index", "will_pit"]

snapshot = {"track_id": "British Grand Prix", "laps_remaining": 18, "current_position": 6,
            "tire_age": 24, "tire_compound": "SOFT", "gap_ahead": 20.4,
            "gap_behind": 1.9, "driver_pace_index": 0.62}

pit_loss = 21.5                                   # British GP, from track_constants.json
snapshot["pit_loss_s"] = pit_loss
snapshot["rejoin_gap_s"] = round(snapshot["gap_ahead"] - pit_loss, 2)   # -1.1 s → rejoins in dirty air

score = lambda will_pit: float(model.predict(pd.DataFrame([{**snapshot, "will_pit": will_pit}],
                                                          columns=FEATURES))[0])
box, stay = score(1), score(0)
print("BOX NOW" if box < stay else "STAY OUT", f"| gain {stay - box:+.2f} positions")
```

## Behaviour that validates the physics

These are the two checks that motivated shipping `pit_loss_s` / `rejoin_gap_s` in the first place — the call tracks *where you land*, not a fixed rule:

- **Monaco** (24.5 s pit loss, high traffic) is scored `STAY OUT` at every tyre age tested — the stop never pays for position on the street.
- **Silverstone** on worn Softs flips from `STAY OUT` to `BOX NOW` as the gap ahead shrinks toward the 21.5 s pit loss, and flips back once `rejoin_gap_s` goes negative enough that the car would rejoin into traffic.

## Training data

**Not real telemetry.** `core/pipeline.py` ships a deterministic physics generator: 20 circuits × 600 snapshots = 12,000 rows. Each row's `final_position` label is produced by an explicit lap-time model — per-compound degradation with a cliff (`SOFT` 15 / `MEDIUM` 22 / `HARD` 30 laps), a dirty-air penalty while stuck behind a rejoin deficit, a clean-air gain when the gap ahead covers the pit loss, and Gaussian outcome noise (σ = 0.7 positions).

The FastF1 miner that would build the same matrix from real results exists in the same module and is reachable with `python -m core.pipeline --real`, but as of 2026 the upstream archive requires an authenticated F1 account, so the artifact published here comes from the generator.

## Evaluation

GroupKFold(3) **grouped by `track_id`** — every fold is scored on circuits the fold never saw, so the numbers describe cross-circuit generalisation rather than memorisation of one track's noise.

| Metric | Value |
| --- | --- |
| CV RMSE | **1.144** positions |
| CV MAE | **0.856** positions |
| Naive baseline MAE (predict `current_position`) | **3.322** positions |
| Training rows / circuits | 12,000 / 20 |
| Hyper-parameters | 400 iterations, LR 0.05, depth 6, RMSE loss, seed 42 |

**Read that honestly.** The model beats the naive baseline by ~74%, but a large part of that is *recovery of the generator's own physics* — the labels are made by a formula, so learning it looks impressive and is not evidence about real races. The useful signal is directional, not absolute: `current_position` (46%) and `will_pit` (23%) dominate, and the rejoin features change decisions in the right places.

## Limitations

- **Simulated labels.** No real race result, no weather, no safety-car window, no tyre-temperature model.
- **One car deep on traffic.** The rejoin penalty considers the car immediately ahead; a queue of three cars behind a slow leader is under-priced.
- **Live gap data is not free.** OpenF1's free endpoints expose positions but not inter-car intervals, so the live feed passes nominal gaps and the counterfactual leans on `current_position` there.
- **Not a decision system.** A portfolio demonstration of a non-blocking inference pipeline. Do not use it to pit a real car.

## Related

- **Hub repo:** https://huggingface.co/Veeda241/tessellate-f1-strategy
- **Architecture & use-case walkthrough (16-page PDF):** https://huggingface.co/Veeda241/tessellate-f1-strategy/resolve/main/Tessellate-AI_Architecture_and_Use_Cases.pdf
- Source code and serving tier: the `Tessellate-AI` repository (FastAPI + zero-dependency dashboard). This card mirrors its `docs/ARCHITECTURE.md` §5–§7.

*Unofficial project. Not affiliated with, endorsed by, or connected to Formula 1, the FIA, or any team.*
