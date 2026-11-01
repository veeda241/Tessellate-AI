import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.jev_scout import analyze_text
from core.config import BRIDGE_PATH, DATA_DIR, FRONTEND_DIR, MODELS_DIR
from core.pipeline import DEFAULT_PIT_LOSS_S, MODEL_FEATURES as FEATURES

# CatBoost predict runs in C++ and releases the GIL, so a thread pool gives
# genuine parallel execution with zero model-serialization cost per call.
executor = ThreadPoolExecutor(max_workers=2)
strategy_model = None
TRACK_MEMORY = {}
MODEL_METADATA = {}


@asynccontextmanager
async def bootstrap_context(app: FastAPI):
    """Lifespan handler: preload the C++ binary and static circuit memory at boot."""
    global strategy_model, TRACK_MEMORY, MODEL_METADATA

    target_binary = MODELS_DIR / "strategy_catboost.cbm"
    if not target_binary.exists():
        raise RuntimeError(f"Missing core runtime model weights binary at {target_binary}")
    strategy_model = CatBoostRegressor()
    strategy_model.load_model(str(target_binary))

    with open(DATA_DIR / "track_constants.json", "r") as f:
        TRACK_MEMORY = json.load(f)

    metadata_path = MODELS_DIR / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            MODEL_METADATA = json.load(f)

    print("⚙️ Inference cluster initialized: native .cbm + in-memory track constants.")
    yield


app = FastAPI(title="Tessellate AI: Non-Blocking Inference Server", lifespan=bootstrap_context)


class InferenceRequest(BaseModel):
    track_name: str
    laps_remaining: int = Field(ge=0, le=78)
    current_position: int = Field(gt=0, le=20)
    tire_age: int = Field(ge=0, le=60)
    tire_compound: str
    radio_transcript: str
    race_control_text: str
    # Race-grade spatial context (optional; realistic defaults when omitted)
    gap_ahead: float = Field(default=1.0, ge=0.0, le=300.0)
    gap_behind: float = Field(default=1.0, ge=0.0, le=300.0)
    driver_pace_index: float = Field(default=90.0, ge=60.0, le=120.0)


def compute_model_score(model_instance: CatBoostRegressor, feature_row: dict) -> float:
    """Synchronous core worker execution loop running inside the isolated CPU pool."""
    frame = pd.DataFrame([feature_row], columns=FEATURES)
    return float(model_instance.predict(frame)[0])


def _build_feature_row(request: InferenceRequest, will_pit: int, pit_loss_s: float) -> dict:
    # Counterfactual convention (matches training labelling): present the SAME observed
    # snapshot for both universes and toggle only `will_pit`. Where you rejoin is *state*,
    # not a decision, so pit_loss_s and rejoin_gap_s are identical in both universes as
    # well — the tree learns how they interact with the flag. We must NOT pre-zero tyre
    # age, pre-add pit loss, or shift position for the box universe here.
    return {
        "track_id": request.track_name,
        "laps_remaining": request.laps_remaining,
        "current_position": request.current_position,
        "tire_age": request.tire_age,
        "tire_compound": request.tire_compound.upper(),
        "gap_ahead": request.gap_ahead,
        "gap_behind": request.gap_behind,
        "pit_loss_s": pit_loss_s,
        "rejoin_gap_s": round(request.gap_ahead - pit_loss_s, 2),
        "driver_pace_index": request.driver_pace_index,
        "will_pit": will_pit,
    }


@app.post("/api/predict_counterfactual")
async def predict_counterfactual(request: InferenceRequest):
    if strategy_model is None:
        raise HTTPException(status_code=500, detail="Inference cluster not initialized.")

    loop = asyncio.get_running_loop()

    track_meta = TRACK_MEMORY.get(request.track_name)
    track_fallback_used = track_meta is None
    pit_loss_s = float(track_meta["pit_loss"]) if track_meta else DEFAULT_PIT_LOSS_S

    # Asynchronously process text features concurrently with tree math (Jev AI scout)
    jev_task = asyncio.create_task(
        analyze_text(request.radio_transcript, request.race_control_text)
    )

    universe_box = _build_feature_row(request, will_pit=1, pit_loss_s=pit_loss_s)
    universe_stay = _build_feature_row(request, will_pit=0, pit_loss_s=pit_loss_s)

    started = time.perf_counter()
    try:
        # Offload intensive prediction matrix operations to prevent event-loop freezing
        future_box = loop.run_in_executor(executor, compute_model_score, strategy_model, universe_box)
        future_stay = loop.run_in_executor(executor, compute_model_score, strategy_model, universe_stay)

        finish_box, finish_stay = await asyncio.gather(future_box, future_stay)
    except Exception as inference_failure:
        if not jev_task.done():
            jev_task.cancel()
        raise HTTPException(
            status_code=500, detail=f"Tree computation pipeline error: {inference_failure}"
        )
    inference_ms = (time.perf_counter() - started) * 1000.0

    jev_indicators = await jev_task
    optimal_move = "BOX" if finish_box < finish_stay else "STAY OUT"

    return {
        "status": "success",
        "jev_scout_indicators": jev_indicators,
        "counterfactual_projections": {
            "BOX_NOW_EXPECTED_FINISH": round(finish_box, 2),
            "STAY_OUT_EXPECTED_FINISH": round(finish_stay, 2),
        },
        "pit_wall_recommendation": optimal_move,
        "expected_position_gain": round(finish_stay - finish_box, 2),
        # Diagnostics only: the rejoin situation the model was shown. Never re-scored.
        "rejoin_context": {
            "pit_loss_s": pit_loss_s,
            "rejoin_gap_s": round(request.gap_ahead - pit_loss_s, 2),
            "rejoins_behind_car_ahead": bool(request.gap_ahead < pit_loss_s),
        },
        "track_fallback_used": track_fallback_used,
        "inference_ms": round(inference_ms, 3),
    }


@app.get("/healthz")
def healthz():
    return {
        "model_loaded": strategy_model is not None,
        "model_version": MODEL_METADATA.get("trained_at", "unknown"),
        "holdout_rmse_positions": MODEL_METADATA.get("holdout_rmse_positions"),
        "holdout_mae_positions": MODEL_METADATA.get("holdout_mae_positions"),
        "baseline_mae_positions": MODEL_METADATA.get("baseline_mae_positions"),
        "total_training_rows": MODEL_METADATA.get("total_training_rows"),
        "track_count": len(TRACK_MEMORY),
    }


@app.get("/api/tracks")
def tracks():
    return {"tracks": sorted(TRACK_MEMORY.keys())}


@app.get("/api/live_state")
def live_state():
    """Serve the ingest streamer's atomic state bridge to the web dashboard.

    The streamer writes this via temp-file + os.replace, so a torn read should
    never occur; we still guard against the file being absent mid-startup.
    """
    try:
        with open(BRIDGE_PATH, "r") as f:
            bridge = json.load(f)
        age = time.time() - BRIDGE_PATH.stat().st_mtime
        return {"available": True, "age_seconds": round(age, 2), **bridge}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # On Windows os.replace can momentarily lock the target; degrade gracefully
        # so the dashboard simply keeps its last-good frame until the next poll.
        return {"available": False}


# Static single-page dashboard (served from the same origin as the API -> no CORS).
# Mounted last so it only handles paths not already matched by the API routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
