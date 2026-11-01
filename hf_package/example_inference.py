"""Minimal consumer for strategy_catboost.cbm — mirrors api/main.py's counterfactual call."""
import json
import pathlib

import pandas as pd
from catboost import CatBoostRegressor

HERE = pathlib.Path(__file__).resolve().parent

FEATURES = [
    "track_id", "laps_remaining", "current_position", "tire_age", "tire_compound",
    "gap_ahead", "gap_behind", "pit_loss_s", "rejoin_gap_s", "driver_pace_index", "will_pit",
]

model = CatBoostRegressor()
model.load_model(str(HERE / "strategy_catboost.cbm"))
tracks = json.loads((HERE / "track_constants.json").read_text())

snapshot = {
    "track_id": "British Grand Prix",
    "laps_remaining": 18,
    "current_position": 6,
    "tire_age": 24,
    "tire_compound": "SOFT",
    "gap_ahead": 20.4,
    "gap_behind": 1.9,
    "driver_pace_index": 0.62,
}

pit_loss = float(tracks[snapshot["track_id"]]["pit_loss"])
snapshot["pit_loss_s"] = pit_loss
snapshot["rejoin_gap_s"] = round(snapshot["gap_ahead"] - pit_loss, 2)

frame = lambda will_pit: pd.DataFrame([{**snapshot, "will_pit": will_pit}], columns=FEATURES)
box = float(model.predict(frame(1))[0])
stay = float(model.predict(frame(0))[0])

call = "BOX NOW" if box < stay else "STAY OUT"
print(f"{snapshot['track_id']} | lap {52 - snapshot['laps_remaining']} | P{snapshot['current_position']} "
      f"on {snapshot['tire_age']}-lap {snapshot['tire_compound']}")
print(f"  projected finish  BOX={box:.2f}  STAY={stay:.2f}")
print(f"  rejoin gap {snapshot['rejoin_gap_s']:+.2f} s (pit loss {pit_loss} s)")
print(f"  call -> {call} ({stay - box:+.2f} positions)")
