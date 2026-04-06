#!/usr/bin/env python3
"""
Weather module for the AI Thermostat Agent.
Primary: Open-Meteo (free, no API key, true daily high/low, hourly resolution).
Fallback: OpenWeatherMap free tier (3-hour intervals).
20-minute cache with stale detection at 6 hours.
"""

import logging
import time
import requests
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_api_key: Optional[str] = None
_lat: Optional[float] = None
_lon: Optional[float] = None
_cache_minutes: int = 20
_stale_hours: int = 6

# Cache (stores normalized format from either source)
_cached_weather: Optional[dict] = None
_cache_timestamp: float = 0

# WMO weather code → condition string (matches OWM condition names)
_WMO_TO_CONDITION = {
    0: "Clear",
    1: "Clear",         # Mainly clear
    2: "Clouds",        # Partly cloudy
    3: "Clouds",        # Overcast
    45: "Mist",         # Fog
    48: "Mist",         # Depositing rime fog
    51: "Drizzle",
    53: "Drizzle",
    55: "Drizzle",
    56: "Drizzle",      # Freezing drizzle
    57: "Drizzle",      # Freezing drizzle heavy
    61: "Rain",
    63: "Rain",
    65: "Rain",         # Heavy rain
    66: "Rain",         # Freezing rain
    67: "Rain",         # Heavy freezing rain
    71: "Snow",
    73: "Snow",
    75: "Snow",
    77: "Snow",         # Snow grains
    80: "Rain",         # Rain showers slight
    81: "Rain",         # Rain showers moderate
    82: "Rain",         # Rain showers violent
    85: "Snow",         # Snow showers slight
    86: "Snow",         # Snow showers heavy
    95: "Thunderstorm",
    96: "Thunderstorm", # Thunderstorm with slight hail
    99: "Thunderstorm", # Thunderstorm with heavy hail
}


@dataclass
class WeatherData:
    current_temp: float       # Fahrenheit
    humidity: int             # percent
    forecast_summary: str     # 1-2 sentence summary for LLM
    is_stale: bool
    alerts: list              # list of alert strings


def init_weather(api_key: str, latitude: float, longitude: float,
                 cache_minutes: int = 20, stale_hours: int = 6):
    """Initialize the weather module."""
    global _api_key, _lat, _lon, _cache_minutes, _stale_hours
    _api_key = api_key
    _lat = latitude
    _lon = longitude
    _cache_minutes = cache_minutes
    _stale_hours = stale_hours
    logger.info("Weather module initialized (lat=%.4f, lon=%.4f)", latitude, longitude)


def get_weather() -> WeatherData:
    """
    Get current weather and forecast.
    Primary: Open-Meteo. Fallback: OpenWeatherMap.
    Returns cached data if within cache window.
    Marks data [STALE] if cache > stale_hours.
    """
    global _cached_weather, _cache_timestamp

    now = time.time()
    cache_age_seconds = now - _cache_timestamp
    cache_age_minutes = cache_age_seconds / 60
    cache_age_hours = cache_age_seconds / 3600

    # Return cached if fresh enough
    if _cached_weather and cache_age_minutes < _cache_minutes:
        return _build_weather_data(_cached_weather, is_stale=False)

    # Try Open-Meteo first
    try:
        normalized = _fetch_open_meteo()
        _cached_weather = normalized
        _cache_timestamp = now
        logger.info("Weather fetched from Open-Meteo")
        return _build_weather_data(_cached_weather, is_stale=False)
    except Exception as e:
        logger.warning("Open-Meteo fetch failed: %s — trying OWM fallback", e)

    # Fallback: OpenWeatherMap
    try:
        normalized = _fetch_owm_normalized()
        _cached_weather = normalized
        _cache_timestamp = now
        logger.info("Weather fetched from OpenWeatherMap (fallback)")
        return _build_weather_data(_cached_weather, is_stale=False)
    except Exception as e:
        logger.error("OWM fallback also failed: %s", e)

    # Return stale cache if we have one
    if _cached_weather:
        is_stale = cache_age_hours > _stale_hours
        return _build_weather_data(_cached_weather, is_stale=is_stale,
                                   stale_hours=cache_age_hours)
    # No cache at all — return defaults
    return WeatherData(
        current_temp=75.0,
        humidity=50,
        forecast_summary="[STALE - weather data unavailable]",
        is_stale=True,
        alerts=[]
    )


def check_weather_alerts(weather: WeatherData) -> list:
    """Check for heat/cold advisories."""
    alerts = []
    if weather.current_temp > 100:
        alerts.append(f"HEAT ADVISORY: Outdoor temp is {weather.current_temp:.0f}F")
    if weather.current_temp < 35:
        alerts.append(f"COLD ADVISORY: Outdoor temp is {weather.current_temp:.0f}F")
    return alerts


@dataclass
class ForecastAnalysis:
    """Pre-digested forecast insights for the agent's directive builder."""
    next_6h_high: float       # Highest temp in next 6 hours
    next_6h_low: float        # Lowest temp in next 6 hours
    next_24h_high: float      # Highest temp in next 24 hours
    next_24h_low: float       # Lowest temp in next 24 hours
    heatwave: bool            # Next 24h high > 95F
    cold_snap: bool           # Next 24h low < 40F
    temp_rising: bool         # Temps trending up over next 6h
    temp_dropping: bool       # Temps trending down over next 6h
    rain_coming: bool         # Rain/storms in next 12h
    hours_to_peak_heat: float # Hours until peak heat (0 if no heatwave)
    hours_to_peak_cold: float # Hours until peak cold (0 if no cold snap)
    pre_cool_now: bool        # True if peak heat is 1-2 hours away
    pre_heat_now: bool        # True if peak cold is 1-2 hours away
    advisory: str             # One-line summary for the directive, or ""


def get_forecast_analysis() -> Optional[ForecastAnalysis]:
    """
    Analyze the cached forecast data and return structured insights.
    Time-aware: uses actual timestamps to slice windows and find peaks.
    Works with both hourly (Open-Meteo) and 3-hour (OWM) resolution.
    Returns None if no forecast data is available.
    """
    if not _cached_weather or "hourly" not in _cached_weather:
        return None

    entries = _cached_weather["hourly"]
    if not entries:
        return None

    now = time.time()

    # Time-based window slicing (works for any resolution)
    next_6h = [e for e in entries if e["dt"] < now + 6 * 3600]
    next_12h = [e for e in entries if e["dt"] < now + 12 * 3600]
    next_24h = [e for e in entries if e["dt"] < now + 24 * 3600]

    # Fallback if windows are empty (e.g. all entries are in the future)
    if not next_6h:
        next_6h = entries[:6]
    if not next_12h:
        next_12h = entries[:12]
    if not next_24h:
        next_24h = entries[:24]

    temps_6h = [e["temp"] for e in next_6h]
    temps_24h = [e["temp"] for e in next_24h]

    next_6h_high = max(temps_6h)
    next_6h_low = min(temps_6h)
    next_24h_high = max(temps_24h)
    next_24h_low = min(temps_24h)

    # Trend: compare first vs last entry in next 6h
    temp_rising = temps_6h[-1] > temps_6h[0] + 3  # Rising by 3+F
    temp_dropping = temps_6h[-1] < temps_6h[0] - 3  # Dropping by 3+F

    heatwave = next_24h_high > 95
    cold_snap = next_24h_low < 40

    rain_coming = any(
        e["condition"] in ("Rain", "Drizzle", "Thunderstorm")
        for e in next_12h
    )

    # ── Time-aware: find WHEN peak temps occur ──
    hours_to_peak_heat = 0.0
    hours_to_peak_cold = 0.0

    if heatwave:
        peak_entry = max(next_24h, key=lambda e: e["temp"])
        peak_dt = peak_entry.get("dt", 0)
        if peak_dt:
            hours_to_peak_heat = max(0, (peak_dt - now) / 3600)

    if cold_snap:
        cold_entry = min(next_24h, key=lambda e: e["temp"])
        cold_dt = cold_entry.get("dt", 0)
        if cold_dt:
            hours_to_peak_cold = max(0, (cold_dt - now) / 3600)

    # Pre-cool/heat window: act 1-2 hours before the event
    pre_cool_now = heatwave and 0 <= hours_to_peak_heat <= 2
    pre_heat_now = cold_snap and 0 <= hours_to_peak_cold <= 2

    # ── Build time-aware advisory ──
    advisory_parts = []
    if heatwave:
        if pre_cool_now:
            advisory_parts.append(
                f"Heatwave peak {next_24h_high:.0f}F in ~{hours_to_peak_heat:.0f}h. Pre-cool the house NOW."
            )
        elif hours_to_peak_heat <= 4:
            advisory_parts.append(
                f"Heatwave peak {next_24h_high:.0f}F in ~{hours_to_peak_heat:.0f}h. Start preparing soon."
            )
        else:
            advisory_parts.append(
                f"Heatwave expected in ~{hours_to_peak_heat:.0f}h (peak {next_24h_high:.0f}F). No action needed yet."
            )

    if cold_snap:
        if pre_heat_now:
            advisory_parts.append(
                f"Cold snap {next_24h_low:.0f}F in ~{hours_to_peak_cold:.0f}h. Keep the house warm NOW."
            )
        elif hours_to_peak_cold <= 4:
            advisory_parts.append(
                f"Cold snap {next_24h_low:.0f}F in ~{hours_to_peak_cold:.0f}h. Prepare soon."
            )
        else:
            advisory_parts.append(
                f"Cold snap expected in ~{hours_to_peak_cold:.0f}h (low {next_24h_low:.0f}F). No action needed yet."
            )

    if temp_rising and not heatwave:
        advisory_parts.append(f"Temps rising to {next_6h_high:.0f}F in next 6h. Consider pre-cooling.")
    if temp_dropping and not cold_snap:
        advisory_parts.append(f"Temps dropping to {next_6h_low:.0f}F in next 6h.")
    if rain_coming:
        advisory_parts.append("Rain/storms expected — temps may drop.")

    return ForecastAnalysis(
        next_6h_high=next_6h_high,
        next_6h_low=next_6h_low,
        next_24h_high=next_24h_high,
        next_24h_low=next_24h_low,
        heatwave=heatwave,
        cold_snap=cold_snap,
        temp_rising=temp_rising,
        temp_dropping=temp_dropping,
        rain_coming=rain_coming,
        hours_to_peak_heat=hours_to_peak_heat,
        hours_to_peak_cold=hours_to_peak_cold,
        pre_cool_now=pre_cool_now,
        pre_heat_now=pre_heat_now,
        advisory=" ".join(advisory_parts),
    )


# ── Fetch: Open-Meteo (primary) ──

def _fetch_open_meteo() -> dict:
    """Fetch weather from Open-Meteo and return normalized format."""
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": _lat,
        "longitude": _lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code",
        "hourly": "temperature_2m,relative_humidity_2m,weather_code,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timeformat": "unixtime",
        "forecast_days": 2,
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    cur = data["current"]
    hourly = data["hourly"]
    daily = data["daily"]

    hourly_entries = []
    for i in range(len(hourly["time"])):
        hourly_entries.append({
            "dt": hourly["time"][i],
            "temp": hourly["temperature_2m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "condition": _WMO_TO_CONDITION.get(hourly["weather_code"][i], "Clouds"),
            "precipitation": hourly["precipitation"][i],
        })

    return {
        "source": "open-meteo",
        "current": {
            "temp": cur["temperature_2m"],
            "humidity": cur["relative_humidity_2m"],
            "condition": _WMO_TO_CONDITION.get(cur["weather_code"], "Clouds"),
        },
        "hourly": hourly_entries,
        "daily": {
            "high": daily["temperature_2m_max"][0],
            "low": daily["temperature_2m_min"][0],
        },
    }


# ── Fetch: OpenWeatherMap (fallback) ──

def _fetch_owm_normalized() -> dict:
    """Fetch from OpenWeatherMap and return normalized format."""
    current = _fetch_current()
    forecast = _fetch_forecast()

    entries = forecast.get("list", [])

    hourly_entries = []
    for e in entries:
        hourly_entries.append({
            "dt": e["dt"],
            "temp": e["main"]["temp"],
            "humidity": e["main"]["humidity"],
            "condition": e["weather"][0]["main"],
            "precipitation": e.get("rain", {}).get("3h", 0) + e.get("snow", {}).get("3h", 0),
        })

    # Daily high/low from first 8 forecast entries (best free tier can do)
    if entries:
        today_temps = [e["main"]["temp"] for e in entries[:8]]
        daily_high = max(today_temps)
        daily_low = min(today_temps)
    else:
        daily_high = current["main"]["temp"]
        daily_low = current["main"]["temp"]

    return {
        "source": "owm",
        "current": {
            "temp": current["main"]["temp"],
            "humidity": current["main"]["humidity"],
            "condition": current["weather"][0]["main"],
        },
        "hourly": hourly_entries,
        "daily": {
            "high": daily_high,
            "low": daily_low,
        },
    }


def _fetch_current() -> dict:
    """Fetch current weather from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    resp = requests.get(url, params={
        "lat": _lat,
        "lon": _lon,
        "appid": _api_key,
        "units": "imperial"
    }, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _fetch_forecast() -> dict:
    """Fetch 5-day forecast from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    resp = requests.get(url, params={
        "lat": _lat,
        "lon": _lon,
        "appid": _api_key,
        "units": "imperial"
    }, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Build output from normalized cache ──

def _build_weather_data(cached: dict, is_stale: bool,
                        stale_hours: float = 0) -> WeatherData:
    """Build WeatherData from normalized cached data."""
    current_temp = cached["current"]["temp"]
    humidity = cached["current"]["humidity"]
    forecast_summary = _build_forecast_summary(cached)

    if is_stale:
        forecast_summary = f"[STALE - last update {stale_hours:.0f} hours ago] {forecast_summary}"

    alerts = []
    if current_temp > 100:
        alerts.append(f"HEAT ADVISORY: {current_temp:.0f}F")
    if current_temp < 35:
        alerts.append(f"COLD ADVISORY: {current_temp:.0f}F")

    return WeatherData(
        current_temp=round(current_temp, 1),
        humidity=humidity,
        forecast_summary=forecast_summary,
        is_stale=is_stale,
        alerts=alerts
    )


def _build_forecast_summary(cached: dict) -> str:
    """Condense forecast into 1-2 sentences for the LLM.
    Uses true daily high/low from the normalized format."""
    hourly = cached.get("hourly", [])
    daily = cached.get("daily", {})

    if not hourly and not daily:
        return "No forecast data available"

    # True daily high/low (main improvement — Open-Meteo provides actual daily extremes)
    high = daily.get("high", 0)
    low = daily.get("low", 0)

    # Dominant weather condition from next 24h of hourly data
    next_24h = hourly[:24] if len(hourly) >= 24 else hourly
    conditions = {}
    for e in next_24h:
        cond = e["condition"]
        conditions[cond] = conditions.get(cond, 0) + 1
    dominant = max(conditions, key=conditions.get) if conditions else "Clear"

    rain_count = sum(1 for e in next_24h
                     if e["condition"] in ("Rain", "Drizzle", "Thunderstorm"))

    summary = f"{dominant.lower()}, high {high:.0f}F low {low:.0f}F"
    if rain_count > 0:
        total = len(next_24h)
        summary += f", rain expected ({rain_count} of next {total} periods)"

    return summary
