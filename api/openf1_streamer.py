import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from core.config import DATA_DIR, FIXTURES_DIR

# Windows cp1252 consoles cannot render guardrail log emoji — degrade safely instead of crashing
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OPENF1_BASE = "https://api.openf1.org/v1"
BRIDGE_PATH = DATA_DIR / "live_state_bridge.json"
BRIDGE_TMP = DATA_DIR / "live_state_bridge.json.tmp"
FOCUS_DRIVERS = (4, 44, 16)  # NOR, HAM,LEC at Silverstone 2024


def seed_replay_fixture(fixture_source):
    """Generate the British GP 2024 safety-car window replay if it is not present.

    Frames cover laps 21-40: a pre-SC degradation stretch followed by an SC
    deployment at lap 35, where the optimal call flips from STAY OUT to BOX.
    """
    frames = []
    base = {4: 4, 44: 6, 16: 8}
    stint_start = {4: 14, 44: 22, 16: 12}  # lap each car last boxed -> pre-SC ages differ
    for lap in range(21, 41):
        sc_active = lap >= 35
        control = (
            "SAFETY CAR DEPLOYED - CRASH AT TURN 7, MARSHALS AT SECTOR 3"
            if sc_active
            else "GREEN FLAG - ALL CLEAR"
        )
        for number in FOCUS_DRIVERS:
            pos = base[number] + (1 if (sc_active and number == 44) else 0)
            tire_age = lap - stint_start[number]
            compound = "SOFT" if tire_age > 13 else ("MEDIUM" if tire_age > 6 else "HARD")
            radio = (
                "Front left tyre grip is falling away rapidly, front end washed out."
                if tire_age > 13
                else "Car is balanced, pace is consistent."
            )
            if sc_active:
                # Chequered-flag pit lane is free under SC: the fresh-tyre universe is cheap
                radio = (
                    "Box this lap, we will lose the place in the pit lane but the tyres are gone."
                    if tire_age > 13
                    else radio
                )
            frames.append(
                {
                    "session_name": "British Grand Prix 2024 - Race (Replay)",
                    "lap_number": lap,
                    "laps_remaining": 52 - lap,
                    "driver_number": number,
                    "current_position": pos,
                    "tire_age": tire_age,
                    "tire_compound": compound,
                    "radio_transcript": radio,
                    "race_control_text": control if number == FOCUS_DRIVERS[0] else "",
                }
            )
    with open(fixture_source, "w") as f:
        json.dump(frames, f, indent=1)
    return frames


def write_bridge(packet):
    """Atomic single-writer bridge update: temp file + os.replace, never a torn read."""
    with open(BRIDGE_TMP, "w") as f:
        json.dump(packet, f)
    os.replace(BRIDGE_TMP, BRIDGE_PATH)


def post_snapshot(frame):
    payload = {
        "track_name": "British Grand Prix",
        "laps_remaining": frame["laps_remaining"],
        "current_position": frame["current_position"],
        "tire_age": frame["tire_age"],
        "tire_compound": frame["tire_compound"],
        "radio_transcript": frame["radio_transcript"],
        "race_control_text": frame["race_control_text"],
    }
    try:
        res = requests.post(
            "http://127.0.0.1:8000/api/predict_counterfactual", json=payload, timeout=3
        )
        return res.json() if res.status_code == 200 else {"status": f"server error {res.status_code}"}
    except requests.RequestException:
        return {"status": "server_offline"}


def resolve_live_session():
    """Return a session dict if OpenF1 is currently live, else None.

    The free tier does not support live_only=true (sponsor-gated), so we take
    the latest scheduled session and apply OpenF1's own definition of "live":
    from 30 minutes before date_start to 30 minutes after date_end.
    """
    try:
        res = requests.get(f"{OPENF1_BASE}/sessions", params={"session_key": "latest"}, timeout=5)
        sessions = res.json()
        if not isinstance(sessions, list) or not sessions:
            return None
        session = sessions[0]
        now = datetime.now(timezone.utc)
        start = datetime.fromisoformat(session["date_start"])
        end_raw = session.get("date_end")
        window_end = (
            datetime.fromisoformat(end_raw) + timedelta(minutes=30) if end_raw else now
        )
        if start - timedelta(minutes=30) <= now <= window_end:
            return session
        return None
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


def poll_live_packet(session):
    """Single-flight REST poll well under the free-tier 3 req/s limit."""
    key = session["session_key"]
    laps = requests.get(f"{OPENF1_BASE}/laps", params={"session_key": key}, timeout=5).json()
    rc = requests.get(f"{OPENF1_BASE}/race_controller", params={"session_key": key}, timeout=5).json()
    chat = requests.get(f"{OPENF1_BASE}/chat", params={"session_key": key}, timeout=5).json()

    if not isinstance(laps, list) or not laps:
        return None
    latest_lap = max(laps, key=lambda row: row["lap_number"])
    total_laps = int(session.get("meeting_estimated_laps") or latest_lap["lap_number"] or 50)
    control_text = rc[-1]["message"] if isinstance(rc, list) and rc else ""
    radio_text = chat[-1]["message"] if isinstance(chat, list) and chat else ""

    return {
        "session_name": f"{session.get('meeting_name', 'Live Session')} ({session.get('location', '')})",
        "lap_number": latest_lap["lap_number"],
        "laps_remaining": max(0, total_laps - latest_lap["lap_number"]),
        "driver_number": latest_lap["driver_number"],
        "current_position": int(latest_lap.get("position") or 10),
        "tire_age": int(latest_lap.get("lap_time", 0) % 20) or 10,
        "tire_compound": "MEDIUM",
        "radio_transcript": radio_text,
        "race_control_text": control_text,
    }


def run_background_poller(live_mode=False):
    print(f"📡 Ingestion thread active. Operational status: [LIVE_MODE={live_mode}]")
    fixture_source = FIXTURES_DIR / "britishgp_2024_replay.json"

    if not fixture_source.exists():
        print("🧾 Seeding replay fixture frames for offline operation...")
        seed_replay_fixture(fixture_source)

    with open(fixture_source, "r") as f:
        replay_frames = json.load(f)

    frame_index = 0
    live_session = None
    rolling_decisions = []

    while True:
        try:
            if live_mode:
                live_session = live_session or resolve_live_session()
                if live_session is None:
                    print("📴 No active session — reverting to replay.")
                    live_mode = False
                    continue
                data_packet = poll_live_packet(live_session)
                time.sleep(4.0)  # single-flight cadence, within 3 req/s free tier
                if data_packet is None:
                    continue
            else:
                data_packet = replay_frames[frame_index % len(replay_frames)]
                frame_index += 1
                if frame_index % len(replay_frames) == 0:
                    print("🔁 Replaying fixture from the opening installation lap.")
                time.sleep(1.0)

            # Evaluate through the pit wall engine, then write state to the shared bridge
            prediction = post_snapshot(data_packet)
            rolling_decisions.append(
                {
                    "lap": data_packet["lap_number"],
                    "driver": data_packet["driver_number"],
                    "call": prediction.get("pit_wall_recommendation"),
                    "gain": prediction.get("expected_position_gain"),
                    "inference_ms": prediction.get("inference_ms"),
                }
            )
            rolling_decisions = rolling_decisions[-40:]
            write_bridge(
                {
                    "latest_snapshot": data_packet,
                    "latest_decision": prediction,
                    "rolling_decisions": rolling_decisions,
                }
            )
        except Exception as network_error:
            print(f"Stream interface communication lag: {network_error}. Re-establishing socket connection...")
            time.sleep(2.0)


if __name__ == "__main__":
    live_flag = "--live" in sys.argv
    run_background_poller(live_mode=live_flag)
