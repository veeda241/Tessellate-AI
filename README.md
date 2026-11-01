# Tessellate AI — F1 Counterfactual Pit Strategy Evaluator

> **BOX NOW or STAY OUT?** Tessellate AI answers the pit-wall's hardest question in milliseconds by simulating both parallel universes of a single race moment — and picking the one that finishes higher.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-009688) ![CatBoost](https://img.shields.io/badge/CatBoost-native%20.cbm-yellow) ![Latency](https://img.shields.io/badge/inference-~5--15ms-brightgreen) ![Status](https://img.shields.io/badge/status-unofficial%20portfolio-orange) [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Veeda241/tessellate-f1-strategy)

An enterprise-shaped, low-latency (<200 ms) motorsport strategy engine: a split-process pipeline combining a **compiled CatBoost C++ inference layer**, a **non-blocking FastAPI** serving tier, an **isolated live/replay ingest** process, and a **zero-dependency web dashboard** — no build step, no Node, no Streamlit.

*A tessellation fills a plane with shapes and no gaps. This model does the same with race distance: fitting the perfect structural tiles (stints) into the total laps.*

**📸 Showcase** — capture the live dashboard into `docs/screenshots/manual_verdict.png` and `docs/screenshots/live_feed.png` (open the app, then **Win + Shift + S**); they render here automatically once committed.

---

## ✨ Highlights

- **Genuine counterfactual inference** — the *same* driver snapshot is scored twice, toggling only the `will_pit` flag; the better projected finish wins.
- **Race-grade ML** — CatBoost regressor with native categorical handling, validated with **GroupKFold by circuit** (measured on *unseen* tracks, not memorized noise).
- **Real engineering constraints, solved** — ONNX categorical-export drift, Windows path/CWD fragility, event-loop blocking, and "no live session at demo time" are each mitigated by design (see [Guardrails](#-the-five-production-guardrails)).
- **Sub-frame latency** — CPU tree math is offloaded to a worker pool via `run_in_executor`, so the async server never blocks; typical core inference is **~5–15 ms**.
- **Self-contained** — runs fully offline on a physics-based generator; swapping in real telemetry is one flag (`--real`).

## 🚀 Quickstart

```powershell
pip install -r requirements.txt
python -m core.pipeline          # build the feature matrix (offline generator by default)
python -m core.train             # train CatBoost -> models/strategy_catboost.cbm + metadata
python -m uvicorn api.main:app --port 8000     # serves BOTH the API and the dashboard
```

Skip the two build steps if you only want to run the server — pull the published weights and circuit memory from the Hub instead:

```powershell
hf download Veeda241/tessellate-f1-strategy strategy_catboost.cbm model_metadata.json --local-dir models
hf download Veeda241/tessellate-f1-strategy track_constants.json --local-dir data
```

Open **http://127.0.0.1:8000** — the dashboard is served by the same process. Swagger UI lives at **/docs**.
To animate the Live feed tab, start the ingest process in a second terminal: `python -m api.openf1_streamer`.

## 📊 At a Glance

| Dimension | Value |
| --- | --- |
| Validation scheme | GroupKFold(3), grouped by `track_id` (unseen-circuit) |
| CV RMSE | **1.14 positions** |
| CV MAE | **0.86 positions** vs naive baseline **3.32** (~74% reduction) |
| Training footprint | 12,000 rows · 20 circuits (offline generator) |
| Core inference latency | ~5–15 ms (target <200 ms) |
| Serving format | Native C++ CatBoost `.cbm` |
| Top learned features | current position ~46%, `will_pit` ~23%, `tire_age` ~9% |

---

## 🏗️ System Architecture

> Full walkthrough — module-by-module reference, the counterfactual contract with measured decisions, request lifecycle, use cases, and the honest limitations found in the code — is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

```
                  ┌────────────────────────────────────────┐
                  │        OpenF1 Live Timing Feed         │
                  │   (REST Poll / Replay JSON Fixture)    │
                  └───────────────────┬────────────────────┘
                                      │ Ingest Process
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       api.openf1_streamer Layer        │
                  │  Owns network I/O; isolates processes  │
                  │  Writes to 'live_state_bridge.json'    │
                  └───────────────────┬────────────────────┘
                                      │ Bridged State JSON
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      FastAPI Server (Event Loop)       │
                  │  Serves API + static web dashboard     │
                  │  Drives async calls without loop blocks│
                  └─────────┬────────────────────┬─────────┘
                            │                    │
          Async Text Ingest │                    │ Offload Tree Math
                            ▼                    ▼
     ┌─────────────────────────────┐      ┌─────────────────────────────┐
     │     Jev AI Text Scout       │      │   Isolated Thread Pool      │
     │  Parses radio / race-control│      │  Runs CatBoost (.cbm) C++   │
     │  text into risk indicators  │      │  inference (GIL released)   │
     └──────────────┬──────────────┘      └──────────────┬──────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      │ Merged Strategic Array
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      Pit Wall Web Dashboard (SPA)      │
                  │  Manual verdicts + live rolling feed   │
                  └────────────────────────────────────────┘
```

## 🧠 Model, Features & Evaluation

- **Model**: CatBoost `CatBoostRegressor` (target = finishing position), served as a native compiled C++ `.cbm`. Categoricals (`track_id`, `tire_compound`) handled natively.
- **Feature schema (11)**: `track_id`, `laps_remaining`, `current_position`, `tire_age`, `tire_compound`, `gap_ahead`, `gap_behind`, `pit_loss_s`, `rejoin_gap_s`, `driver_pace_index`, `will_pit`. The two pit columns are what make the *traffic after the stop* explicit: `pit_loss_s` is the circuit's stationary pit-lane delta and `rejoin_gap_s = gap_ahead − pit_loss_s` says whether the stop hands the place back or lands you in clean air.
- **Validation — GroupKFold(3) by circuit**: every fold is scored on entirely unseen tracks, so the metric reflects generalization rather than memorizing one circuit's noise.
- **Metrics** (in `models/model_metadata.json`): CV RMSE ≈ **1.14**, CV MAE ≈ **0.86** vs a naive "predict current position" baseline of **3.32** (~74% reduction).
- **Behavior check**: the engine **stays out at Monaco** at every tyre age probed (a 24.5 s pit lane plus a queue that it cannot clear), flips **STAY OUT → BOX at Silverstone between 20 and 24 laps of Soft wear**, and boxes sooner the further the car ahead has escaped. Calls are circuit- and traffic-aware, not a fixed rule.

## 🔬 Counterfactual Contract & API

Both universes receive the **identical observed state** (position, tyre age, compound, gaps, pace, pit-lane delta, rejoin gap) and differ **only in `will_pit`**. Where you rejoin is *state*, not a decision, so `pit_loss_s` and `rejoin_gap_s` are the same on both sides and the tree learns how they interact with the flag. The server does **not** pre-add traffic or zero tyre age at inference time — doing so would double-count what the labels already teach.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web dashboard (same origin as API — no CORS) |
| `POST` | `/api/predict_counterfactual` | Evaluate BOX_NOW vs STAY_OUT for a snapshot |
| `GET` | `/api/live_state` | Latest bridge snapshot, decision & rolling feed |
| `GET` | `/api/tracks` | Circuits held in track memory |
| `GET` | `/healthz` | Model-loaded flag, version, metrics, track count |

The prediction response carries both projections, the Jev AI indicators (`safety_car_escalation_risk`, `tire_urgency_label`), `expected_position_gain`, a `rejoin_context` block (`pit_loss_s`, `rejoin_gap_s`, `rejoins_behind_car_ahead`) showing the traffic situation the model was shown, and `inference_ms` for latency verification.

## 🌐 Web Dashboard

A zero-dependency single-page app (vanilla HTML/CSS/JS + a hand-rolled Canvas chart) served by the FastAPI process. Two tabs: **Manual Snapshot Evaluation** (form → verdict banner, twin projection cards, pit-lane delta and rejoin gap with a plain-English read of what the stop costs, safety-car risk meter, latency) and **Live Pit Wall Feed** (polls `/api/live_state`, animates rolling BOX/STAY-OUT margins with a per-frame rejoin column as the replay runs each car's stint into its own tyre cliff).

## 🏛️ The Five Production Guardrails

| Production Risk | System Mitigation |
| --- | --- |
| **Data-source / version mismatch** | Event-level `try/except` per Grand Prix; if the mined matrix falls below threshold, the pipeline injects a calibrated fallback so the build never breaks. |
| **CatBoost ONNX export drift** | The server relies entirely on native `.cbm`. ONNX export is isolated in a `try/except` that warns but never blocks (ONNX-ML can't export categoricals). |
| **No live session at demo time** | Recorded replay fixtures are the primary path; live mode is a `--live` flag that auto-reverts to replay when nothing is on track. |
| **Windows path / CWD issues** | `core/config.py` resolves an immutable `PROJECT_ROOT` pathlib anchor; every module derives paths from it. |
| **Event-loop blocking by core math** | Ingest is a separate process writing the bridge atomically (`os.replace`); FastAPI offloads CatBoost inference to a worker pool via `run_in_executor`. |

## 📉 Honest Status & Data-Sourcing Reality

- **Real telemetry is one flag away, but gated.** The `core.pipeline` FastF1 miner is real, yet in 2026 its upstream live-timing archive requires an authenticated F1 account and the retired Ergast API breaks the older `2.3.x` line. Without that access, `python -m core.pipeline` falls back to an **offline physics-based generator** (20 circuits, tyre-degradation + pit-loss + clean-air dynamics). Feature contract, trainer and serving path are identical — run `python -m core.pipeline --real` on an authenticated machine for genuine multi-season data.
- **The Jev AI text scout is currently a deterministic keyword prototype** (safety-car / tyre-urgency tiers) exposed behind an **LLM-ready async interface** — it is honest scaffolding, not yet a trained language model.
- **Read the MAE as physics recovery, not race accuracy.** The rejoin-traffic columns are inputs to the same generator that writes the labels, so most of the step from 1.48 → 0.86 positions is the model being handed mechanisms it previously had to guess at. It is a cleaner, better-validated result *about that data*; it is still not a claim of pit-wall accuracy against a real Sunday.
- **Live traffic is not observable yet.** OpenF1's free endpoints carry no time intervals, so `gap_ahead`/`gap_behind` are flat placeholders in `--live` mode; the traffic-after-the-stop term only does work in the replay fixture and the manual tab until an interval feed is wired in.

## 🎯 Built for the F1 Paddock

Tessellate AI is an **independent, unofficial learning project** inspired by the pit-wall strategy work performed across the sport. It is built for fans, analysts and engineers of the teams on the **2026 Formula 1 grid**:

**Red Bull Racing · Ferrari · Mercedes · McLaren · Aston Martin · Alpine · Williams · Racing Bulls · Haas · Audi · Cadillac**

> ⚠️ **Disclaimer:** This project is **not affiliated with, endorsed by, or connected to** Formula 1, the FIA, or any team or manufacturer named above. All names, marks and team identities are the property of their respective owners and are used here descriptively only.

## 🗺️ Roadmap

- Wire a real lightweight classifier/LLM into the Jev AI scout (interface already async & isolated).
- Historical back-testing harness (evaluate counterfactual calls against actual results).
- Live WebSocket/MQTT streaming ingest (OpenF1 sponsor tier) for true race-weekend latency — and real `gap_ahead`/`gap_behind` intervals to replace the live placeholders.
- Safety-car pit loss: a deployed SC roughly halves the stationary delta, which is exactly when the rejoin term flips. Needs the SC flag to reach the feature row instead of only the text scout.
- Expand features: multi-car pit-queue arrival, stint-plan optimiser, driver-to-driver pace differentials.

## 📁 Repository Layout

```
tessellate-ai/
├── data/
│   ├── track_constants.json             # Circuit parameters
│   ├── live_state_bridge.json           # Atomic single-writer feed (generated)
│   └── fixtures/britishgp_2024_replay.json  # SC-window replay (auto-seeded)
├── core/
│   ├── config.py                        # PROJECT_ROOT absolute path anchor
│   ├── pipeline.py                      # FastF1 miner (--real) + offline generator
│   └── train.py                         # CatBoost .cbm training, GroupKFold, metadata
├── api/
│   ├── main.py                          # Non-blocking server: API + static dashboard
│   ├── jev_scout.py                      # Async text scout (LLM-ready interface)
│   └── openf1_streamer.py                # Separated replay/live ingest process
├── ui/                                  # Zero-dependency web dashboard
│   ├── index.html · styles.css · app.js
└── models/
    ├── strategy_catboost.cbm            # Production C++ inference binary (generated)
    └── model_metadata.json              # Schema, metrics, versions (generated)
```

## 🧱 Tech Stack

`Python 3.11` · `FastAPI` + `uvicorn` · `CatBoost` (native `.cbm`) · `pandas` / `numpy` · `Pydantic v2` (validation) · `ThreadPoolExecutor` + `run_in_executor` (non-blocking inference) · vanilla JS + Canvas UI.

## 🤝 Contact & License

Built as a portfolio project — feedback from anyone in motorsport strategy or ML infrastructure is welcome. See [LICENSE](LICENSE) for terms.
