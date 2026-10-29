import asyncio

# Keyword tiers for the System-One style text scout. Interface-shaped so a real
# LLM backend can be swapped in behind the same async contract later.
TIRE_KEYWORDS_CRITICAL = ("falling away", "flat spot", "losing it", "vibration", "destroyed")
TIRE_KEYWORDS_ELEVATED = ("grip", "degrad", "washed out", "sliding", "overheating", "graining")
SC_KEYWORDS = ("safety car", "sc deployed", "crash", "debris", "incident")


def _tire_urgency(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in TIRE_KEYWORDS_CRITICAL):
        return "Critical"
    if any(k in lowered for k in TIRE_KEYWORDS_ELEVATED):
        return "Elevated"
    return "Low"


async def analyze_text(radio_transcript: str, race_control_text: str) -> dict:
    """Asynchronous unstructured-text evaluation (simulated Jev AI roundtrip)."""
    await asyncio.sleep(0.04)

    combined = f"{radio_transcript} {race_control_text}".upper()
    safety_car_risk = 0.95 if "SAFETY CAR" in combined else (
        0.55 if any(k.upper() in combined for k in SC_KEYWORDS[1:]) else 0.05
    )

    return {
        "safety_car_escalation_risk": safety_car_risk,
        "tire_urgency_label": _tire_urgency(radio_transcript),
    }
