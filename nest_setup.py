#!/usr/bin/env python3
"""
Nest SDM API Setup & Test Script
Run this to exchange your auth code for tokens and list your devices.
"""

import json
import requests
import os

TOKENS_FILE = "nest_tokens.json"

def get_input(prompt):
    return input(prompt).strip()

def exchange_code():
    """Step 1: Exchange authorization code for tokens."""
    print("=" * 50)
    print("STEP 1: Exchange auth code for tokens")
    print("=" * 50)

    client_id = get_input("Paste your OAuth Client ID: ")
    client_secret = get_input("Paste your OAuth Client Secret: ")
    auth_code = get_input("Paste your authorization code: ")
    project_id = get_input("Paste your Device Access Project ID: ")

    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": auth_code,
        "grant_type": "authorization_code",
        "redirect_uri": "https://www.google.com"
    })

    if resp.status_code != 200:
        print(f"\nERROR: {resp.status_code}")
        print(resp.text)
        return None

    tokens = resp.json()

    # Save everything we need
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "project_id": project_id,
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
    }

    with open(TOKENS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nTokens saved to {TOKENS_FILE}")
    print("  access_token: YES")
    print("  refresh_token: YES")
    return data


def refresh_access_token(data):
    """Refresh the access token using the refresh token."""
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": data["client_id"],
        "client_secret": data["client_secret"],
        "refresh_token": data["refresh_token"],
        "grant_type": "refresh_token",
    })

    if resp.status_code != 200:
        print(f"Refresh failed: {resp.status_code}")
        print(resp.text)
        return None

    new_token = resp.json()["access_token"]
    data["access_token"] = new_token

    with open(TOKENS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return new_token


def list_devices(data):
    """Step 2: List all Nest devices."""
    print("\n" + "=" * 50)
    print("STEP 2: Listing your Nest devices")
    print("=" * 50)

    url = f"https://smartdevicemanagement.googleapis.com/v1/enterprises/{data['project_id']}/devices"
    headers = {"Authorization": f"Bearer {data['access_token']}"}

    resp = requests.get(url, headers=headers)

    if resp.status_code == 401:
        print("Access token expired, refreshing...")
        new_token = refresh_access_token(data)
        if not new_token:
            return
        headers = {"Authorization": f"Bearer {new_token}"}
        resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code}")
        print(resp.text)
        return

    devices = resp.json().get("devices", [])

    if not devices:
        print("No devices found! Check that you authorized the correct home.")
        return

    print(f"\nFound {len(devices)} device(s):\n")

    for device in devices:
        device_id = device["name"].split("/")[-1]
        device_type = device.get("type", "unknown")
        traits = device.get("traits", {})

        print(f"  Device ID: {device_id}")
        print(f"  Type: {device_type}")

        # Temperature
        temp_trait = traits.get("sdm.devices.traits.Temperature", {})
        if temp_trait:
            celsius = temp_trait.get("ambientTemperatureCelsius", "?")
            if isinstance(celsius, (int, float)):
                fahrenheit = celsius * 9/5 + 32
                print(f"  Indoor Temp: {celsius}C / {fahrenheit:.1f}F")

        # Humidity
        humidity_trait = traits.get("sdm.devices.traits.Humidity", {})
        if humidity_trait:
            print(f"  Humidity: {humidity_trait.get('ambientHumidityPercent', '?')}%")

        # Mode
        mode_trait = traits.get("sdm.devices.traits.ThermostatMode", {})
        if mode_trait:
            print(f"  Mode: {mode_trait.get('mode', '?')}")

        # Setpoint
        setpoint_trait = traits.get("sdm.devices.traits.ThermostatTemperatureSetpoint", {})
        if setpoint_trait:
            cool = setpoint_trait.get("coolCelsius")
            heat = setpoint_trait.get("heatCelsius")
            if cool:
                print(f"  Cool Target: {cool}C / {cool * 9/5 + 32:.1f}F")
            if heat:
                print(f"  Heat Target: {heat}C / {heat * 9/5 + 32:.1f}F")

        # HVAC status
        hvac_trait = traits.get("sdm.devices.traits.ThermostatHvac", {})
        if hvac_trait:
            print(f"  HVAC Status: {hvac_trait.get('status', '?')}")

        print()

    # Save device ID for later use
    if devices:
        data["device_id"] = devices[0]["name"]
        with open(TOKENS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved device ID to {TOKENS_FILE}")


def test_set_temperature(data):
    """Step 3: Test setting a temperature."""
    print("\n" + "=" * 50)
    print("STEP 3: Test temperature control")
    print("=" * 50)

    if "device_id" not in data:
        print("No device ID saved. Run device listing first.")
        return

    temp_f = get_input("Enter target temperature in F (or 'skip'): ")
    if temp_f.lower() == "skip":
        print("Skipped.")
        return

    temp_c = (float(temp_f) - 32) * 5 / 9

    # Detect current mode to use correct command
    url = f"https://smartdevicemanagement.googleapis.com/v1/{data['device_id']}"
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    resp = requests.get(url, headers=headers)

    if resp.status_code == 401:
        new_token = refresh_access_token(data)
        if not new_token:
            return
        headers = {"Authorization": f"Bearer {new_token}"}
        resp = requests.get(url, headers=headers)

    traits = resp.json().get("traits", {})
    mode = traits.get("sdm.devices.traits.ThermostatMode", {}).get("mode", "COOL")

    if mode == "COOL":
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetCool"
        params = {"coolCelsius": round(temp_c, 1)}
    elif mode == "HEAT":
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetHeat"
        params = {"heatCelsius": round(temp_c, 1)}
    else:
        print(f"Current mode is {mode} — setting to COOL first")
        mode_resp = requests.post(
            f"https://smartdevicemanagement.googleapis.com/v1/{data['device_id']}:executeCommand",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "command": "sdm.devices.commands.ThermostatMode.SetMode",
                "params": {"mode": "COOL"}
            }
        )
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetCool"
        params = {"coolCelsius": round(temp_c, 1)}

    resp = requests.post(
        f"https://smartdevicemanagement.googleapis.com/v1/{data['device_id']}:executeCommand",
        headers={**headers, "Content-Type": "application/json"},
        json={"command": command, "params": params}
    )

    if resp.status_code == 200:
        print(f"\nSUCCESS: Temperature set to {temp_f}F ({temp_c:.1f}C)")
    else:
        print(f"\nERROR: {resp.status_code}")
        print(resp.text)


if __name__ == "__main__":
    print("NEST SDM API SETUP & TEST")
    print("=" * 50)

    # Check for existing tokens
    if os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE) as f:
            data = json.load(f)
        print(f"Found existing tokens in {TOKENS_FILE}")
        choice = get_input("Use existing tokens? (y/n): ")
        if choice.lower() != "y":
            data = exchange_code()
    else:
        data = exchange_code()

    if not data:
        print("Setup failed.")
        exit(1)

    list_devices(data)
    test_set_temperature(data)

    print("\n" + "=" * 50)
    print("SETUP COMPLETE")
    print("=" * 50)
    print(f"Tokens saved to: {TOKENS_FILE}")
    print("Your agent can now use these to control your thermostat.")
