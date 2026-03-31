#!/usr/bin/env python3
"""
Weather module for the AI Thermostat Agent.
Uses OpenWeatherMap free tier (current + 5-day forecast).
20-minute cache with stale detection at 6 hours.
"""

import logging
import time
import requests
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_api_key: Optional[str] = None
_lat: Optional[float] = None
_lon: Optional[float] = None
_cache_minutes: int = 20
_stale_hours: int = 6

# Cache
_cached_weather: Optional[dict] = None
_cache_timestamp: float = 0


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

    # Try to fetch fresh data
    try:
        current = _fetch_current()
        forecast = _fetch_forecast()
        _cached_weather = {"current": current, "forecast": forecast}
        _cache_timestamp = now
        return _build_weather_data(_cached_weather, is_stale=False)
    except Exception as e:
        logger.error("Weather fetch failed: %s", e)
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
    Time-aware: calculates WHEN peak temps arrive and only advises
    pre-cooling/heating 1-2 hours before the event.
    Returns None if no forecast data is available.
    """
    if not _cached_weather or "forecast" not in _cached_weather:
        return None

    entries = _cached_weather["forecast"].get("list", [])
    if not entries:
        return None

    now = time.time()

    # Next 6 hours (2 entries), 12 hours (4), 24 hours (8)
    next_6h = entries[:2] if len(entries) >= 2 else entries
    next_12h = entries[:4] if len(entries) >= 4 else entries
    next_24h = entries[:8] if len(entries) >= 8 else entries

    temps_6h = [e["main"]["temp"] for e in next_6h]
    temps_24h = [e["main"]["temp"] for e in next_24h]

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
        e["weather"][0]["main"] in ("Rain", "Drizzle", "Thunderstorm")
        for e in next_12h
    )

    # ── Time-aware: find WHEN peak temps occur ──
    hours_to_peak_heat = 0.0
    hours_to_peak_cold = 0.0

    if heatwave:
        # Find the entry with the highest temp and its timestamp
        peak_entry = max(next_24h, key=lambda e: e["main"]["temp"])
        peak_dt = peak_entry.get("dt", 0)
        if peak_dt:
            hours_to_peak_heat = max(0, (peak_dt - now) / 3600)

    if cold_snap:
        # Find the entry with the lowest temp and its timestamp
        cold_entry = min(next_24h, key=lambda e: e["main"]["temp"])
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


def _build_weather_data(cached: dict, is_stale: bool,
                        stale_hours: float = 0) -> WeatherData:
    """Build WeatherData from cached API responses."""
    current = cached["current"]
    forecast = cached["forecast"]

    current_temp = current["main"]["temp"]
    humidity = current["main"]["humidity"]
    forecast_summary = _build_forecast_summary(forecast)

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


def _build_forecast_summary(forecast: dict) -> str:
    """Condense the 5-day forecast into 1-2 sentences for the LLM."""
    entries = forecast.get("list", [])
    if not entries:
        return "No forecast data available"

    # Look at next 24 hours (8 x 3-hour intervals)
    next_24h = entries[:8]

    temps = [e["main"]["temp"] for e in next_24h]
    high = max(temps)
    low = min(temps)

    # Get dominant weather condition
    conditions = {}
    for e in next_24h:
        desc = e["weather"][0]["main"]
        conditions[desc] = conditions.get(desc, 0) + 1
    dominant = max(conditions, key=conditions.get)

    # Check for rain
    rain_count = sum(1 for e in next_24h
                     if e["weather"][0]["main"] in ("Rain", "Drizzle", "Thunderstorm"))

    summary = f"{dominant.lower()}, high {high:.0f}F low {low:.0f}F"
    if rain_count > 0:
        summary += f", rain expected ({rain_count} of next 8 periods)"

    return summary
