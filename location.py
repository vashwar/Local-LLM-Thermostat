#!/usr/bin/env python3
"""
OwnTracks location module for the AI Thermostat Agent.
Tracks phone locations via MQTT and manages vacation mode.
Pattern follows weather.py / nest_api.py: module-level state, init function, getters.
"""

import asyncio
import json
import logging
import math
import time
from typing import Optional, Callable

import database

logger = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────
_enabled: bool = False
_broker_host: str = "localhost"
_broker_port: int = 1883
_broker_user: Optional[str] = None
_broker_pass: Optional[str] = None
_topics: list = []
_geofence_miles: float = 30.0
_stale_hours: float = 12.0
_vacation_temp_min: int = 60
_vacation_temp_max: int = 85
_vacation_eval_interval_minutes: int = 60

_home_lat: float = 0.0
_home_lon: float = 0.0

_phone_locations: dict = {}  # topic -> {lat, lon, miles_from_home, last_update}
_vacation_mode: bool = False
_vacation_change_callback: Optional[Callable] = None
_initialized: bool = False


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
    """Initialize location tracking from config. Call once at startup."""
    global _enabled, _broker_host, _broker_port, _broker_user, _broker_pass
    global _topics, _geofence_miles, _stale_hours
    global _vacation_temp_min, _vacation_temp_max, _vacation_eval_interval_minutes
    global _home_lat, _home_lon, _initialized

    ot = config.get("owntracks", {})
    _enabled = ot.get("enabled", False)

    if not _enabled:
        logger.info("OwnTracks location tracking disabled")
        return

    _broker_host = ot.get("broker_host", "localhost")
    _broker_port = ot.get("broker_port", 1883)
    _broker_user = ot.get("broker_user")
    _broker_pass = ot.get("broker_pass")
    _topics = ot.get("topics", [])
    _geofence_miles = ot.get("geofence_miles", 30.0)
    _stale_hours = ot.get("stale_hours", 12.0)
    _vacation_temp_min = ot.get("vacation_temp_min", 60)
    _vacation_temp_max = ot.get("vacation_temp_max", 85)
    _vacation_eval_interval_minutes = ot.get("vacation_eval_interval_minutes", 60)

    # Home coordinates from weather config
    weather_cfg = config.get("weather", {})
    _home_lat = weather_cfg.get("latitude", 0.0)
    _home_lon = weather_cfg.get("longitude", 0.0)

    _initialized = True
    logger.info("OwnTracks initialized: %d topics, geofence=%.0f mi, home=(%.4f, %.4f)",
                len(_topics), _geofence_miles, _home_lat, _home_lon)


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


def get_phone_locations() -> dict:
    """Return current phone location data for status display."""
    return dict(_phone_locations)


def set_vacation_change_callback(fn: Callable):
    """Register an async callback for vacation mode transitions. fn(is_vacation: bool)."""
    global _vacation_change_callback
    _vacation_change_callback = fn


def _on_mqtt_message(topic: str, payload: bytes):
    """Process an incoming OwnTracks MQTT message."""
    global _phone_locations

    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Invalid JSON from %s", topic)
        return

    # OwnTracks location messages have _type: "location"
    if data.get("_type") != "location":
        return

    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return

    miles = _haversine(_home_lat, _home_lon, lat, lon)

    _phone_locations[topic] = {
        "lat": lat,
        "lon": lon,
        "miles_from_home": round(miles, 1),
        "last_update": time.time(),
    }

    logger.info("Location update [%s]: (%.4f, %.4f) — %.1f mi from home",
                topic, lat, lon, miles)

    try:
        database.log_location_event("location_update", topic, lat, lon, miles, _vacation_mode)
    except Exception:
        pass  # DB not initialized (e.g. in tests)

    _evaluate_vacation_mode()


def _evaluate_vacation_mode():
    """Evaluate whether vacation mode should change based on current phone locations."""
    global _vacation_mode

    if not _topics:
        return

    now = time.time()
    stale_threshold = _stale_hours * 3600

    phones_away = 0
    phones_home = 0
    phones_stale = 0
    phones_with_data = 0

    for topic in _topics:
        loc = _phone_locations.get(topic)
        if loc is None:
            # No data for this phone — counts as neither away nor home
            continue

        phones_with_data += 1
        age = now - loc["last_update"]

        if age > stale_threshold:
            phones_stale += 1
            phones_home += 1  # Stale = assume home (safer default)
            continue

        if loc["miles_from_home"] > _geofence_miles:
            phones_away += 1
        else:
            phones_home += 1

    # Fail-safe: if no phone has any data at all, vacation stays OFF
    if phones_with_data == 0:
        return

    old_mode = _vacation_mode

    if phones_home > 0:
        # Any phone home → vacation OFF
        _vacation_mode = False
    elif phones_away > 0 and phones_home == 0:
        # All non-stale phones are away → vacation ON
        _vacation_mode = True
    # else: all stale, keep current

    if _vacation_mode != old_mode:
        logger.info("Vacation mode changed: %s → %s (away=%d, home=%d, stale=%d)",
                    old_mode, _vacation_mode, phones_away, phones_home, phones_stale)
        try:
            database.log_location_event("vacation_change", "", 0, 0, 0, _vacation_mode)
        except Exception:
            pass  # DB not initialized (e.g. in tests)
        if _vacation_change_callback:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(_vacation_change_callback(_vacation_mode))
                else:
                    loop.run_until_complete(_vacation_change_callback(_vacation_mode))
            except Exception as e:
                logger.error("Vacation change callback failed: %s", e)


async def mqtt_loop():
    """Async task: connect to Mosquitto, subscribe to OwnTracks topics, process messages."""
    if not _enabled:
        logger.info("OwnTracks MQTT loop disabled — exiting")
        return

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        logger.error("paho-mqtt not installed — OwnTracks disabled. pip install paho-mqtt")
        return

    while True:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

            if _broker_user:
                client.username_pw_set(_broker_user, _broker_pass)

            def on_connect(client, userdata, flags, reason_code, properties=None):
                if reason_code == 0:
                    logger.info("Connected to MQTT broker %s:%d", _broker_host, _broker_port)
                    for topic in _topics:
                        client.subscribe(topic)
                        logger.info("Subscribed to %s", topic)
                else:
                    logger.error("MQTT connection failed: %s", reason_code)

            def on_message(client, userdata, msg):
                logger.info("MQTT raw message on %s (%d bytes)", msg.topic, len(msg.payload))
                _on_mqtt_message(msg.topic, msg.payload)

            def on_disconnect(client, userdata, flags, reason_code, properties=None):
                logger.warning("MQTT disconnected (reason=%s), will retry...", reason_code)

            client.on_connect = on_connect
            client.on_message = on_message
            client.on_disconnect = on_disconnect

            client.connect(_broker_host, _broker_port, keepalive=60)
            client.loop_start()

            # Wait for connection to establish
            await asyncio.sleep(2)

            # Keep running until disconnected
            while client.is_connected():
                await asyncio.sleep(5)

            client.loop_stop()

        except Exception as e:
            logger.error("MQTT error: %s — retrying in 30s", e)

        # Retry delay
        await asyncio.sleep(30)
