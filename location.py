#!/usr/bin/env python3
"""
ARP-based presence detection for the AI Thermostat Agent.
Scans the Windows ARP table for known phone MAC addresses to determine
if anyone is home. Manages vacation mode based on consecutive misses.
Pattern follows weather.py / nest_api.py: module-level state, init function, getters.
"""

import asyncio
import logging
import math
import subprocess
import time
from typing import Optional, Callable

import database

logger = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────
_enabled: bool = False
_phone_macs: list = []
_vacation_temp_min: int = 60
_vacation_temp_max: int = 85
_vacation_eval_interval_minutes: int = 60

_vacation_mode: bool = False
_vacation_change_callback: Optional[Callable] = None
_initialized: bool = False

# ARP presence state
_consecutive_misses: int = 0
_required_misses: int = 2


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two lat/lon points using the haversine formula."""
    R = 3958.8  # Earth radius in miles
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def init_location(config: dict):
    """Initialize presence detection from config. Call once at startup."""
    global _enabled, _phone_macs, _required_misses
    global _vacation_temp_min, _vacation_temp_max, _vacation_eval_interval_minutes
    global _initialized, _consecutive_misses

    presence = config.get("presence", {})
    _phone_macs = [mac.lower() for mac in presence.get("phone_macs", [])]
    _required_misses = presence.get("consecutive_misses", 2)
    _enabled = len(_phone_macs) > 0

    if not _enabled:
        logger.info("ARP presence detection disabled (no phone_macs configured)")
        return

    # Vacation settings (can come from presence or owntracks for backward compat)
    ot = config.get("owntracks", {})
    _vacation_temp_min = presence.get("vacation_temp_min", ot.get("vacation_temp_min", 60))
    _vacation_temp_max = presence.get("vacation_temp_max", ot.get("vacation_temp_max", 85))
    _vacation_eval_interval_minutes = presence.get(
        "vacation_eval_interval_minutes",
        ot.get("vacation_eval_interval_minutes", 60)
    )

    _consecutive_misses = 0
    _initialized = True
    logger.info("ARP presence initialized: %d MACs, required_misses=%d",
                len(_phone_macs), _required_misses)


def is_vacation_mode() -> bool:
    """Return whether vacation mode is currently active. Safe to call before init."""
    return _vacation_mode


def get_vacation_temp_min() -> int:
    """Return the vacation mode minimum temperature."""
    return _vacation_temp_min


def get_vacation_temp_max() -> int:
    """Return the vacation mode maximum temperature."""
    return _vacation_temp_max


def get_vacation_eval_interval_minutes() -> int:
    """Return the evaluation interval in minutes during vacation mode."""
    return _vacation_eval_interval_minutes


def set_vacation_change_callback(fn: Callable):
    """Register an async callback for vacation mode transitions. fn(is_vacation: bool)."""
    global _vacation_change_callback
    _vacation_change_callback = fn


def _run_arp() -> str:
    """Run 'arp -a' and return its stdout. Separated for testability."""
    result = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=10)
    return result.stdout


def _parse_arp_ip_for_mac(arp_output: str, mac: str) -> str | None:
    """Find the IP address associated with a MAC in the ARP output. Returns None if not found."""
    mac_stripped = mac.replace(":", "").replace("-", "").lower()
    for line in arp_output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            line_mac = parts[1].replace("-", "").replace(":", "").lower()
            if line_mac == mac_stripped:
                return parts[0]
    return None


def _ping_check(ip: str) -> bool:
    """Ping an IP 3 times. Returns True if any reply is received."""
    try:
        result = subprocess.run(
            ["ping", "-n", "3", "-w", "1000", ip],
            capture_output=True, text=True, timeout=10
        )
        # Windows ping returns 0 if any reply received
        return result.returncode == 0
    except Exception:
        return False


def check_arp_presence():
    """
    Check if known phones are actually on the network. Called by agent each eval cycle.

    1. Read ARP table for known MACs to find their IPs
    2. Ping each found IP to verify the device is actually reachable
       (stale ARP cache entries will fail the ping)
    3. Any phone reachable → someone home → vacation OFF (immediate)
    4. No phones reachable → increment consecutive misses
    5. Required consecutive misses reached → vacation ON
    """
    global _consecutive_misses

    if not _enabled:
        return

    try:
        arp_output = _run_arp()
    except Exception as e:
        logger.error("ARP scan failed: %s", e)
        return

    # Find IPs for known MACs, then ping to verify they're actually reachable
    reachable_macs = []
    for mac in _phone_macs:
        ip = _parse_arp_ip_for_mac(arp_output, mac)
        if ip and _ping_check(ip):
            reachable_macs.append(mac)
            logger.debug("ARP: %s (%s) is reachable", mac, ip)
        elif ip:
            logger.debug("ARP: %s (%s) in cache but ping failed — stale entry", mac, ip)

    if reachable_macs:
        # Someone is home
        _consecutive_misses = 0
        logger.info("ARP: %d/%d phone(s) reachable on network", len(reachable_macs), len(_phone_macs))

        try:
            database.log_location_event("arp_detected", ",".join(reachable_macs), 0, 0, 0, _vacation_mode)
        except Exception:
            pass

        _set_vacation(False)
    else:
        # No phones reachable
        _consecutive_misses += 1
        logger.info("ARP: no phones reachable (miss %d/%d)", _consecutive_misses, _required_misses)

        try:
            database.log_location_event("arp_miss", f"miss_{_consecutive_misses}", 0, 0, 0, _vacation_mode)
        except Exception:
            pass

        if _consecutive_misses >= _required_misses:
            _set_vacation(True)


def _set_vacation(active: bool):
    """Set vacation mode and fire callback if changed."""
    global _vacation_mode

    old_mode = _vacation_mode
    _vacation_mode = active

    if _vacation_mode != old_mode:
        logger.info("Vacation mode changed: %s → %s", old_mode, _vacation_mode)
        try:
            database.log_location_event("vacation_change", "", 0, 0, 0, _vacation_mode)
        except Exception:
            pass
        if _vacation_change_callback:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(_vacation_change_callback(_vacation_mode))
                else:
                    loop.run_until_complete(_vacation_change_callback(_vacation_mode))
            except Exception as e:
                logger.error("Vacation change callback failed: %s", e)
