---
title: Tessellate AI — F1 Pit Strategy
emoji: 🏎️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
short_description: BOX NOW or STAY OUT, scored twice on one snapshot
tags:
  - catboost
  - fastapi
  - formula-1
  - strategy
---

# Tessellate AI — counterfactual pit-strategy engine

Answers **BOX NOW or STAY OUT** by scoring the *same* race snapshot twice and toggling only `will_pit`, then recommending the universe that finishes higher. The 2026 version carries `pit_loss_s` and `rejoin_gap_s` so the call accounts for **where the car actually rejoins** — behind the car it was chasing, or in clean air.

Two tabs: **Manual Snapshot Evaluation** (enter position, tyre age, compound, gaps → verdict, twin projections, rejoin context, latency) and **Live Pit Wall Feed** (a recorded Silverstone replay streamed through the same engine, one frame per second).

Machine-readable weights, schema and metrics: [Veeda241/tessellate-f1-strategy](https://huggingface.co/Veeda241/tessellate-f1-strategy). Architecture walkthrough PDF: [resolve link](https://huggingface.co/Veeda241/tessellate-f1-strategy/resolve/main/Tessellate-AI_Architecture_and_Use_Cases.pdf). Source: [github.com/veeda241/Tessellate-AI](https://github.com/veeda241/Tessellate-AI).

## Honest about what it learned

- Labels come from an **offline physics generator** (12,000 rows across 20 circuits: per-compound tyre degradation with a cliff, pit-lane loss, dirty-air and clean-air dynamics) — **not** real Formula 1 telemetry. Real data is one flag away (`python -m core.pipeline --real`) but gated behind an authenticated F1 account.
- CV MAE 0.856 positions vs a naive "predict current position" baseline of 3.322, measured with **GroupKFold grouped by circuit** (scored on unseen tracks). A large share of that margin is recovery of the generator's own formula, so treat the numbers as directional, not as evidence about real races.
- The traffic model is **one car deep**: it prices the rejoin deficit against the car directly ahead, not a queue behind a slow leader.
- The "Jev AI" text scout is a deterministic keyword classifier, not a language model.
- Not a decision system. Do not pit a real car on this.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/predict_counterfactual` | BOX vs STAY for one snapshot; returns both projections, `rejoin_context`, `expected_position_gain`, `inference_ms` |
| `GET` | `/api/live_state` | Latest replay frame, decision and rolling feed |
| `GET` | `/api/tracks` | Circuits in track memory with their pit-lane deltas |
| `GET` | `/healthz` | Model-loaded flag, metrics, track count |
| `GET` | `/docs` | Swagger UI |

This Space runs the documented two-process design inside one container: uvicorn serves the API and dashboard, and a second process replays a recorded British GP through it. Free CPU tier — the first visit after idle can take a minute to build and boot.

*Unofficial project. Not affiliated with, endorsed by, or connected to Formula 1, the FIA, or any team.*
