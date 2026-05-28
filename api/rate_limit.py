"""In-memory daily rate limiting for the answer endpoints.

Two layers:
  - Global daily cost cap (USD). Protects the OpenRouter key from a
    catastrophic bill if someone hammers the endpoint.
  - Per-IP daily query count. Stops one bad actor from single-handedly
    burning the global cap.

State is in-memory. Render free tier loses memory on cold start (~15min
idle), but that's acceptable: cold starts happen during quiet periods,
so the worst-case loss is one day's tracking during low traffic.

The reserve-then-record pattern handles concurrent requests: we reserve
an estimated cost up front (so requests can't all sneak past the cap
simultaneously), then settle with the real usage after the model returns.
"""

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, Request


DAILY_COST_CAP_USD = float(os.environ.get("DAILY_COST_CAP_USD", "25"))
PER_IP_DAILY_LIMIT = int(os.environ.get("PER_IP_DAILY_LIMIT", "50"))

# Conservative pre-reservation estimate. Real cost usually lands lower
# (~$0.02), but reserving more keeps us safely under the cap when many
# requests are in flight at once. Settled to actual cost in record_actual.
ESTIMATED_COST_PER_REQUEST_USD = 0.05


_lock = threading.Lock()
_state = {
    "date": None,        # UTC date string (YYYY-MM-DD)
    "reserved": 0.0,     # USD reserved by in-flight requests
    "spent": 0.0,        # USD settled from completed requests
    "per_ip": {},        # ip -> query count (counts attempts, not just successes)
}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _maybe_roll_day() -> None:
    """Reset counters if the UTC date has changed. Caller must hold _lock."""
    today = _today()
    if _state["date"] != today:
        if _state["date"] is not None:
            # Log the prior day's totals before rolling over.
            print(
                f"[rate_limit] day rollover {_state['date']} -> {today}: "
                f"spent=${_state['spent']:.4f}, "
                f"unique_ips={len(_state['per_ip'])}",
                flush=True,
            )
        _state["date"] = today
        _state["reserved"] = 0.0
        _state["spent"] = 0.0
        _state["per_ip"] = {}


def get_client_ip(request: Request) -> str:
    """Best-effort client IP. Render sits behind a proxy, so X-Forwarded-For
    is the user; falls back to request.client.host for local dev."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        # X-Forwarded-For can be a comma-separated chain; the first entry
        # is the original client.
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_and_reserve(ip: str) -> None:
    """Raise HTTPException(429) if either cap would be exceeded. Otherwise
    reserve an estimated cost slot for this request."""
    with _lock:
        _maybe_roll_day()

        # Per-IP check first — cheaper to reject and more informative.
        ip_count = _state["per_ip"].get(ip, 0)
        if ip_count >= PER_IP_DAILY_LIMIT:
            print(
                f"[rate_limit] per-IP cap hit: ip={ip} count={ip_count} "
                f"limit={PER_IP_DAILY_LIMIT}",
                flush=True,
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "per_ip_cap_exceeded",
                    "message": (
                        "You've reached today's question limit on this site. "
                        "You can bring your own OpenRouter key to keep going."
                    ),
                    "byok_supported": True,
                },
            )

        # Global cost check — count both already-spent and currently-reserved.
        projected = _state["spent"] + _state["reserved"] + ESTIMATED_COST_PER_REQUEST_USD
        if projected > DAILY_COST_CAP_USD:
            print(
                f"[rate_limit] global cap hit: "
                f"spent=${_state['spent']:.4f} reserved=${_state['reserved']:.4f} "
                f"cap=${DAILY_COST_CAP_USD}",
                flush=True,
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "global_cap_exceeded",
                    "message": (
                        "Today's shared question quota for this site is used "
                        "up. You can bring your own OpenRouter key to keep "
                        "going — your key, your spend."
                    ),
                    "byok_supported": True,
                },
            )

        # Reserve slot.
        _state["reserved"] += ESTIMATED_COST_PER_REQUEST_USD
        _state["per_ip"][ip] = ip_count + 1


def record_actual(actual_cost_usd: float) -> None:
    """Settle a previously-reserved slot with the real cost. Releases the
    reservation and adds to the spent total."""
    with _lock:
        _maybe_roll_day()
        _state["reserved"] = max(0.0, _state["reserved"] - ESTIMATED_COST_PER_REQUEST_USD)
        _state["spent"] += max(0.0, actual_cost_usd)
        print(
            f"[rate_limit] settled: +${actual_cost_usd:.5f}, "
            f"day total=${_state['spent']:.4f}/{DAILY_COST_CAP_USD}",
            flush=True,
        )


def release_reservation() -> None:
    """Release a reservation without recording any spend. Used when the
    upstream call failed before producing usage data."""
    with _lock:
        _maybe_roll_day()
        _state["reserved"] = max(0.0, _state["reserved"] - ESTIMATED_COST_PER_REQUEST_USD)


def get_quota_status() -> dict:
    """Return current cap usage for the public /quota endpoint. Exposes
    a percentage, not the dollar amount, to avoid signalling the cap size."""
    with _lock:
        _maybe_roll_day()
        used = _state["spent"] + _state["reserved"]
        pct = min(1.0, used / DAILY_COST_CAP_USD) if DAILY_COST_CAP_USD > 0 else 0.0
        return {
            "percent_used": round(pct, 3),
            "byok_supported": True,
        }
