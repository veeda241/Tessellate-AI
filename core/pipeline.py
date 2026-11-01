import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from core.config import DATA_DIR

# Windows cp1252 consoles cannot render guardrail log emoji — degrade safely instead of crashing
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

np.random.seed(42)

OUTPUT_COLUMNS = [
    "track_id",
    "laps_remaining",
    "current_position",
    "tire_age",
    "compound_degradation",
    "tire_compound",
    "gap_ahead",
    "gap_behind",
    "pit_loss_s",
    "rejoin_gap_s",
    "driver_pace_index",
    "will_pit",
    "final_position",
]
# The canonical model contract: everything the label matrix carries except the target
# and the generator-internal wear column. core/train.py and api/main.py both import this,
# so the training schema and the serving schema cannot drift apart silently.
MODEL_FEATURES = [c for c in OUTPUT_COLUMNS if c not in ("compound_degradation", "final_position")]

# Per-compound wear rates: seconds of lap-time loss accumulated per lap of tyre age.
COMPOUND_DEGRADATION = {"SOFT": 0.085, "MEDIUM": 0.050, "HARD": 0.032}

# Rejoin-traffic physics. Pitting surrenders the circuit's pit-lane loss, so what you
# pay afterwards depends on who you land behind and who catches you while you are
# stuck in their wake. These constants turn that into positions.
CLEAR_RATE_S_PER_LAP = 0.6   # per-lap deficit fresh tyres erase against a worn car ahead
DIRTY_AIR_S_PER_LAP = 0.25   # lap time lost sitting in the car ahead's dirty air
AVG_INTERCAR_GAP_S = 1.5     # typical midfield spacing: seconds per place gained/lost
REJOIN_LEAP_PLACE = 0.8      # the place you hand to the car you were attacking
REJOIN_TRAFFIC_MAX_PLACES = 2.5  # cap: a rejoin costs places, it does not lap you
DEFAULT_PIT_LOSS_S = 21.0    # prior for a circuit with no measured pit-lane loss

# Tyre wear is not linear: a compound holds, then falls off a cliff. Without the
# quadratic term staying out on 30-lap Softs looks almost free, which makes "never
# box" look optimal — a degenerate strategy no pit wall recognises. The cliff lap is
# per compound, because that is exactly how the tyres differ in a race.
COMPOUND_CLIFF_LAP = {"SOFT": 15, "MEDIUM": 22, "HARD": 30}
WEAR_CLIFF_K = 0.015         # s/lap per lap^2 past the compound's cliff

# Circuit catalogue: pit_loss (s), traffic_index (places lost joining the queue),
# and typical total race laps. Values are realistic motorsport priors.
CIRCUITS = {
    "Bahrain Grand Prix": (22.0, 3, 57),
    "Saudi Arabian Grand Prix": (21.0, 3, 50),
    "Australian Grand Prix": (19.5, 3, 58),
    "Japanese Grand Prix": (20.5, 3, 53),
    "Chinese Grand Prix": (21.5, 3, 56),
    "Miami Grand Prix": (21.0, 4, 57),
    "Emilia Romagna Grand Prix": (21.5, 3, 63),
    "Monaco Grand Prix": (24.5, 6, 78),
    "Canadian Grand Prix": (17.0, 4, 70),
    "Spanish Grand Prix": (19.0, 3, 66),
    "Austrian Grand Prix": (17.5, 3, 71),
    "British Grand Prix": (21.5, 2, 52),
    "Hungarian Grand Prix": (22.0, 3, 70),
    "Belgian Grand Prix": (18.0, 2, 44),
    "Dutch Grand Prix": (18.5, 4, 72),
    "Italian Grand Prix": (21.5, 3, 53),
    "Singapore Grand Prix": (24.0, 4, 62),
    "United States Grand Prix": (22.5, 3, 56),
    "Mexico Grand Prix": (21.0, 3, 71),
    "Abu Dhabi Grand Prix": (20.0, 3, 58),
}

# Guardrail 1 threshold: the offline generator always clears this, so the synthetic
# path only trips when the real FastF1 extractor is not run or comes up empty.
FALLBACK_THRESHOLD = 500


# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA PATH (FastF1). Requires an authenticated F1 account in 2026; invoke
# with `python -m core.pipeline --real`. Kept identical in feature contract to
# the offline generator so a single model serves both.
# ─────────────────────────────────────────────────────────────────────────────
def _final_position_map(session):
    results = session.results
    col = "DriverAbbreviation" if "DriverAbbreviation" in results.columns else "Abbreviation"
    mapping = {}
    for _, r in results.iterrows():
        pos = r["Position"]
        if pd.isna(pos):
            continue
        try:
            mapping[r[col]] = int(pos)
        except (ValueError, TypeError):
            continue
    return mapping


def _pace_index_map(laps):
    pace = {}
    for driver, d_laps in laps.groupby("Driver"):
        try:
            quick = d_laps.pick_quicklaps()
        except Exception:
            quick = d_laps
        if quick.empty or quick["LapTime"].isna().all():
            continue
        pace[driver] = float(quick["LapTime"].median().total_seconds())
    return pace


def _pit_lap_sets(laps):
    pitted = laps[laps["PitInTime"].notna()]
    return {d: set(int(x) for x in rows["LapNumber"]) for d, rows in pitted.groupby("Driver")}


def _gap_arrays(gap_to_leader_sorted):
    gl = np.asarray(gap_to_leader_sorted, dtype=float)
    n = len(gl)
    ahead = np.zeros(n)
    behind = np.zeros(n)
    for j in range(n):
        ahead[j] = 0.0 if j == 0 else (gl[j] - gl[j - 1] if np.isfinite(gl[j] - gl[j - 1]) and gl[j] >= gl[j - 1] else 1.5)
        behind[j] = 0.0 if j == n - 1 else (gl[j + 1] - gl[j] if np.isfinite(gl[j + 1] - gl[j]) and gl[j + 1] >= gl[j] else 1.5)
    return ahead, behind


def extract_race_grade_features(seasons=(2023, 2024)):
    try:
        import fastf1
    except Exception as exc:
        print(f"❌ FastF1 unavailable for real mining: {exc}")
        return []

    cache_dir = DATA_DIR / "fastf1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(str(cache_dir))
    except Exception:
        pass

    print(f"🏁 Starting real data mining pipeline for seasons: {list(seasons)}...")
    rows = []
    for year in seasons:
        try:
            schedule = fastf1.get_event_schedule(year)
            race_events = schedule[~schedule["EventFormat"].str.lower().isin(["testing", "test"])]
            for _, race in race_events.iterrows():
                try:
                    session = fastf1.get_session(year, race["EventName"], "R")
                    session.load(telemetry=False, weather=False)
                    laps = session.laps.copy()
                    laps = laps[laps["Position"].notna()]
                    max_lap = int(laps["LapNumber"].max())
                    track_id = str(race["EventName"])
                    track_pit_loss = CIRCUITS.get(track_id, (DEFAULT_PIT_LOSS_S,))[0]
                    final_map = _final_position_map(session)
                    pace_map = _pace_index_map(laps)
                    pit_sets = _pit_lap_sets(laps)
                    for lap_num, lap_data in laps.groupby("LapNumber"):
                        lap_num = int(lap_num)
                        if lap_num >= max_lap:
                            continue
                        ordered = lap_data.sort_values("Position").reset_index(drop=True)
                        ahead, behind = _gap_arrays(ordered["GapToLeader"].values)
                        for j, row in ordered.iterrows():
                            driver = row["Driver"]
                            if driver not in final_map or pd.isna(row["TyreLife"]):
                                continue
                            compound = str(row["Compound"]).upper()
                            pitted_next = (lap_num + 1) in pit_sets.get(driver, ())
                            rows.append(
                                {
                                    "track_id": track_id,
                                    "laps_remaining": int(max_lap - lap_num),
                                    "current_position": int(row["Position"]),
                                    "tire_age": int(row["TyreLife"]),
                                    "compound_degradation": COMPOUND_DEGRADATION.get(compound, 0.05),
                                    "tire_compound": compound,
                                    "gap_ahead": float(ahead[j]),
                                    "gap_behind": float(behind[j]),
                                    "pit_loss_s": float(track_pit_loss),
                                    "rejoin_gap_s": round(float(ahead[j]) - track_pit_loss, 2),
                                    "driver_pace_index": float(pace_map.get(driver, 90.0)),
                                    "will_pit": 1 if pitted_next else 0,
                                    "final_position": final_map[driver],
                                }
                            )
                    print(f"  ✅ {year} {track_id}: running total {len(rows)} rows")
                except Exception as session_err:
                    print(f"⚠️ Skipped {year} {race['EventName']}: {session_err}")
                    continue
        except Exception as schedule_err:
            print(f"❌ Failed to reach schedule registry for season {year}: {schedule_err}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE PHYSICS-BASED GENERATOR. Encodes tyre degradation, pit loss, clean-air
# recovery and driver pace so every rich feature carries signal and the
# counterfactual decision varies by circuit (Monaco stays out, Silverstone boxes).
# ─────────────────────────────────────────────────────────────────────────────
def generate_realistic_features(n_rows=12000):
    print("🧪 Building offline physics-based race telemetry matrix...")
    rng = np.random.default_rng(42)
    track_names = list(CIRCUITS.keys())
    tracks = rng.choice(track_names, size=n_rows)
    compounds = rng.choice(["SOFT", "MEDIUM", "HARD"], size=n_rows, p=[0.45, 0.35, 0.20])

    pit_loss = np.array([CIRCUITS[t][0] for t in tracks])
    traffic = np.array([CIRCUITS[t][1] for t in tracks])
    total_laps = np.array([CIRCUITS[t][2] for t in tracks])
    degrade = np.array([COMPOUND_DEGRADATION[c] for c in compounds])

    laps_remaining = rng.integers(3, 48, size=n_rows)
    current_position = rng.integers(1, 21, size=n_rows)
    tire_age = rng.integers(2, 32, size=n_rows)
    gap_ahead = np.round(rng.exponential(1.2, size=n_rows) + 0.2, 3)
    gap_behind = np.round(rng.exponential(1.2, size=n_rows) + 0.2, 3)
    # A race is not always a train. Roughly one snapshot in five sits in a strung-out
    # field where the car ahead is 15-45 s away — the only regime in which a pit stop
    # can land you in clean air. Without this tail the rejoin decision has no range and
    # the model can never learn the undercut, because it never sees one.
    strung_ahead = rng.random(n_rows) < 0.20
    gap_ahead[strung_ahead] = np.round(rng.uniform(15.0, 45.0, int(strung_ahead.sum())), 3)
    strung_behind = rng.random(n_rows) < 0.12
    gap_behind[strung_behind] = np.round(rng.uniform(10.0, 35.0, int(strung_behind.sum())), 3)
    driver_pace_index = np.round(rng.normal(90.0, 1.3, size=n_rows), 3)  # median clean-air lap time
    pace_skill = (90.0 - driver_pace_index)  # faster driver -> positive places edge over the race
    will_pit = rng.integers(0, 2, size=n_rows)

    # Fraction of the race still to run drives how much a tyre advantage can be cashed in
    recovery_window = laps_remaining / total_laps

    # Tyre wear is not linear. A compound behaves predictably until it crosses its cliff,
    # after which lap time falls away quadratically — the effect every pit wall actually
    # reacts to. Without the cliff, staying out on a 30-lap-old Soft costs almost the same
    # as a 20-lap one, no fresh tyre can ever repay the stop, and the engine degenerates
    # into "always hold track position".
    past_cliff = np.maximum(0.0, tire_age - np.array([COMPOUND_CLIFF_LAP[c] for c in compounds]))
    wear = tire_age * degrade + WEAR_CLIFF_K * past_cliff**2       # seconds lost per lap

    stay_out_loss = wear * recovery_window * 3.0     # cars catch you as you fade
    # A distant car behind eases the pressure, but only up to a point: once nobody is
    # within a few seconds there is no more relief to be had, and an unbounded term here
    # would swamp the rejoin cost and invert the traffic signal we are modelling.
    stay_out_loss -= 0.4 * np.clip(gap_behind - 1.0, 0.0, 6.0)
    box_gain = wear * recovery_window * 2.2          # fresh rubber lets you pass worn cars
    box_cost = traffic + 0.15 * (pit_loss - 20.0)                  # a longer pit lane costs more, not less

    # Where you rejoin. Pitting hands back pit_loss seconds, so the car you were
    # chasing arrives ahead of you unless your gap to them already covered that loss.
    # Landing behind them costs that place, and every lap spent in their wake lets the
    # car behind close up on you too — that is the traffic the fresh tyres must repay.
    # Bounded, because a rejoin costs you places, not laps: the cars you lose to are
    # limited by the field, and fresh rubber takes them back over a stint.
    rejoin_gap = gap_ahead - pit_loss                              # >0: undercut works into clean air
    deficit = np.maximum(0.0, -rejoin_gap)                         # s stuck behind the car ahead
    stuck_laps = np.minimum(laps_remaining, deficit / CLEAR_RATE_S_PER_LAP)
    dirty_air_s = stuck_laps * DIRTY_AIR_S_PER_LAP
    rejoin_traffic = np.minimum(
        REJOIN_TRAFFIC_MAX_PLACES,
        REJOIN_LEAP_PLACE * (deficit > 0)
        + np.maximum(0.0, dirty_air_s - gap_behind) / AVG_INTERCAR_GAP_S * 0.6,
    )

    final = np.where(
        will_pit == 1,
        current_position + box_cost - box_gain + pace_skill + rejoin_traffic,
        # Clean air eases the stayer's degradation, but it is bounded: a 40 s gap ahead
        # is worth the same breathing room as a 10 s one.
        current_position + stay_out_loss + pace_skill
        - 0.10 * np.clip(gap_ahead - 1.0, 0.0, 10.0),
    )
    final = final + rng.normal(0, 0.7, size=n_rows)               # residual race randomness
    final_position = np.clip(np.round(final), 1, 20).astype(int)

    df = pd.DataFrame(
        {
            "track_id": tracks,
            "laps_remaining": laps_remaining,
            "current_position": current_position,
            "tire_age": tire_age,
            "compound_degradation": degrade,
            "tire_compound": compounds,
            "gap_ahead": gap_ahead,
            "gap_behind": gap_behind,
            "pit_loss_s": np.round(pit_loss, 2),
            "rejoin_gap_s": np.round(rejoin_gap, 2),
            "driver_pace_index": driver_pace_index,
            "will_pit": will_pit,
            "final_position": final_position,
        }
    )
    print(f"🧪 Generated {len(df)} simulated rows across {df['track_id'].nunique()} circuits")
    return df.to_dict("records")


def _write(rows):
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(DATA_DIR / "historical_sc_features.csv", index=False)
    print(f"📦 Pipeline finalized. Wrote {len(df)} race telemetry rows to historical_sc_features.csv")
    return df


def write_track_constants():
    """Publish the serving-side circuit memory from the catalogue the labels were built
    with, so the API can never disagree with the training physics about a circuit name.
    pit_loss_s is a model feature, which makes this file load-bearing, not decorative.
    """
    payload = {
        name: {"pit_loss": vals[0], "traffic_index": vals[1], "total_laps": vals[2]}
        for name, vals in CIRCUITS.items()
    }
    out = DATA_DIR / "track_constants.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"🗺️ Circuit memory written to {out} ({len(payload)} tracks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tessellate AI feature mining pipeline")
    parser.add_argument("--real", action="store_true", help="Mine real data via FastF1 (needs F1 account creds)")
    args = parser.parse_args()

    rows = []
    if args.real:
        rows = extract_race_grade_features()

    if len(rows) < FALLBACK_THRESHOLD:
        if args.real:
            print(f"🚨 Real mining returned {len(rows)} rows (upstream archive unreachable / unauthenticated).")
        print("🧬 Falling back to the offline physics-based generator.")
        rows = generate_realistic_features()

    _write(rows)
    write_track_constants()
