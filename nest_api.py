#!/usr/bin/env python3
"""
Nest SDM API wrapper for the AI Thermostat Agent.
Supports multiple named thermostats.
"""

import json
import logging
import requests
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_tokens: Optional[dict] = None
_tokens_path: Optional[str] = None
_devices: list = []  # [{"name": "Upstairs Bedroom", "device_id": "..."}]
_last_known_targets: dict = {}  # {device_id: last_target_F}


@dataclass
class ThermostatState:
    name: str                # "Upstairs Bedroom", "Downstairs Kitchen"
    device_id: str           # full SDM device ID
    indoor_temp: float       # Fahrenheit
    humidity: int            # percent
    mode: str                # "cooling", "heating", "off"
    target_temp: Optional[float]  # Fahrenheit
    hvac_running: bool       # True if actively heating/cooling


def init_nest(tokens_path: str, devices: list = None):
    """Initialize the Nest API module by loading tokens and device list."""
    global _tokens, _tokens_path, _devices
    _tokens_path = tokens_path
    with open(tokens_path) as f:
        _tokens = json.load(f)

    if devices:
        _devices = devices
    elif "device_id" in _tokens:
        # Fallback: single device from nest_tokens.json
        _devices = [{"name": "Thermostat", "device_id": _tokens["device_id"]}]

    logger.info("Nest API initialized (%d device(s): %s)",
                len(_devices), ", ".join(d["name"] for d in _devices))


def get_all_thermostat_states() -> list:
    """Get state for all configured thermostats."""
    states = []
    for device in _devices:
        try:
            state = get_thermostat_state(device["device_id"], device["name"])
            states.append(state)
        except Exception as e:
            logger.error("Failed to get state for %s: %s", device["name"], e)
    return states


def get_thermostat_state(device_id: str = None, name: str = None) -> ThermostatState:
    """Get current thermostat state from Nest SDM API."""
    _ensure_valid_token()

    # Default to first device if not specified
    if device_id is None:
        device_id = _devices[0]["device_id"]
        name = _devices[0]["name"]
    if name is None:
        name = _device_name(device_id)

    url = f"https://smartdevicemanagement.googleapis.com/v1/{device_id}"
    headers = {"Authorization": f"Bearer {_tokens['access_token']}"}

    resp = requests.get(url, headers=headers, timeout=15)

    if resp.status_code == 401:
        _refresh_token()
        headers = {"Authorization": f"Bearer {_tokens['access_token']}"}
        resp = requests.get(url, headers=headers, timeout=15)

    resp.raise_for_status()
    traits = resp.json().get("traits", {})

    # Temperature (Celsius -> Fahrenheit)
    temp_c = traits.get("sdm.devices.traits.Temperature", {}).get(
        "ambientTemperatureCelsius", 72)
    indoor_temp = round(temp_c * 9 / 5 + 32, 1)

    # Humidity
    humidity = traits.get("sdm.devices.traits.Humidity", {}).get(
        "ambientHumidityPercent", 50)

    # Mode
    mode_raw = traits.get("sdm.devices.traits.ThermostatMode", {}).get("mode", "OFF")
    mode_map = {"COOL": "cooling", "HEAT": "heating", "HEATCOOL": "auto", "OFF": "off"}
    mode = mode_map.get(mode_raw, mode_raw.lower())

    # Target temperature
    setpoint = traits.get("sdm.devices.traits.ThermostatTemperatureSetpoint", {})
    target_temp = None
    if mode_raw == "COOL" and "coolCelsius" in setpoint:
        target_temp = round(setpoint["coolCelsius"] * 9 / 5 + 32, 1)
    elif mode_raw == "HEAT" and "heatCelsius" in setpoint:
        target_temp = round(setpoint["heatCelsius"] * 9 / 5 + 32, 1)
    elif mode_raw == "HEATCOOL":
        cool_c = setpoint.get("coolCelsius")
        heat_c = setpoint.get("heatCelsius")
        if cool_c and heat_c:
            target_temp = round(((cool_c + heat_c) / 2) * 9 / 5 + 32, 1)

    # HVAC running status
    hvac_status = traits.get("sdm.devices.traits.ThermostatHvac", {}).get("status", "OFF")
    hvac_running = hvac_status in ("COOLING", "HEATING")

    return ThermostatState(
        name=name,
        device_id=device_id,
        indoor_temp=indoor_temp,
        humidity=humidity,
        mode=mode,
        target_temp=target_temp,
        hvac_running=hvac_running
    )


def set_temperature(temp_f: float, device_id: str = None) -> bool:
    """Set the thermostat target temperature (Fahrenheit). Returns True on success."""
    _ensure_valid_token()

    if device_id is None:
        device_id = _devices[0]["device_id"]

    temp_c = round((temp_f - 32) * 5 / 9, 1)

    # Get current mode to use correct command
    state = get_thermostat_state(device_id)

    if state.mode == "cooling":
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetCool"
        params = {"coolCelsius": temp_c}
    elif state.mode == "heating":
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetHeat"
        params = {"heatCelsius": temp_c}
    elif state.mode == "off":
        if not set_mode("COOL", device_id):
            return False
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetCool"
        params = {"coolCelsius": temp_c}
    else:
        logger.warning("Unsupported mode for set_temperature: %s", state.mode)
        return False

    url = f"https://smartdevicemanagement.googleapis.com/v1/{device_id}:executeCommand"
    headers = {
        "Authorization": f"Bearer {_tokens['access_token']}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, headers=headers, json={
        "command": command,
        "params": params
    }, timeout=15)

    if resp.status_code == 401:
        _refresh_token()
        headers["Authorization"] = f"Bearer {_tokens['access_token']}"
        resp = requests.post(url, headers=headers, json={
            "command": command,
            "params": params
        }, timeout=15)

    if resp.status_code == 200:
        _last_known_targets[device_id] = temp_f
        name = _device_name(device_id)
        logger.info("[%s] Temperature set to %.1fF (%.1fC)", name, temp_f, temp_c)
        return True
    else:
        logger.error("Failed to set temperature: %s %s", resp.status_code, resp.text)
        return False


def set_mode(mode: str, device_id: str = None) -> bool:
    """Set the thermostat mode (COOL, HEAT, OFF). Returns True on success."""
    _ensure_valid_token()

    if device_id is None:
        device_id = _devices[0]["device_id"]

    url = f"https://smartdevicemanagement.googleapis.com/v1/{device_id}:executeCommand"
    headers = {
        "Authorization": f"Bearer {_tokens['access_token']}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, headers=headers, json={
        "command": "sdm.devices.commands.ThermostatMode.SetMode",
        "params": {"mode": mode}
    }, timeout=15)

    if resp.status_code == 401:
        _refresh_token()
        headers["Authorization"] = f"Bearer {_tokens['access_token']}"
        resp = requests.post(url, headers=headers, json={
            "command": "sdm.devices.commands.ThermostatMode.SetMode",
            "params": {"mode": mode}
        }, timeout=15)

    if resp.status_code == 200:
        logger.info("Mode set to %s", mode)
        return True
    else:
        logger.error("Failed to set mode: %s %s", resp.status_code, resp.text)
        return False


def detect_manual_override(device_id: str, current_target: Optional[float]) -> bool:
    """Detect if someone manually changed the thermostat."""
    last = _last_known_targets.get(device_id)

    if last is None or current_target is None:
        _last_known_targets[device_id] = current_target
        return False

    if abs(current_target - last) > 0.5:
        name = _device_name(device_id)
        logger.info("[%s] Manual override detected: %.1fF -> %.1fF", name, last, current_target)
        _last_known_targets[device_id] = current_target
        return True

    return False


def _device_name(device_id: str) -> str:
    """Look up friendly name for a device ID."""
    for d in _devices:
        if d["device_id"] == device_id:
            return d["name"]
    return "Unknown"


def _ensure_valid_token():
    """Ensure we have a valid access token, refresh if needed."""
    if not _tokens or "access_token" not in _tokens:
        raise RuntimeError("Nest API not initialized — call init_nest() first")


def _refresh_token():
    """Refresh the access token using the refresh token."""
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": _tokens["client_id"],
        "client_secret": _tokens["client_secret"],
        "refresh_token": _tokens["refresh_token"],
        "grant_type": "refresh_token",
    }, timeout=15)

    if resp.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {resp.status_code} {resp.text}")

    _tokens["access_token"] = resp.json()["access_token"]

    with open(_tokens_path, "w") as f:
        json.dump(_tokens, f, indent=2)

    logger.info("Access token refreshed and saved")
