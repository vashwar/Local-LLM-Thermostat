#!/usr/bin/env python3
"""
AI Thermostat Agent — The Brain
Autonomous climate control using a local LLM (Qwen 4B via llama.cpp).
"""

import asyncio
import json
import logging
import re
import sys
import time
import requests
import yaml
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Fix __main__ double-import: when run as "python agent.py", this module
# is loaded as "__main__". But telegram_bot.py does "import agent" which
# creates a SECOND copy. This ensures both names point to the same object,
# so set_telegram_send_fn() updates the variable agent_loop() reads.
if __name__ == "__main__":
    sys.modules["agent"] = sys.modules[__name__]

import database
import weather
import nest_api
import llm_server

logger = logging.getLogger(__name__)

# ── Hard-coded guardrails (not configurable) ──────────────────────
TEMP_MIN = 65
TEMP_MAX = 80
MAX_CHANGES_PER_HOUR = 6
MANUAL_OVERRIDE_BACKOFF_MINUTES = 120
MODE_TRANSITION_GAP_MINUTES = 5

# ── System prompt — compact for Qwen 4B ──────────────────────────
SYSTEM_PROMPT = """You are a thermostat agent for the {zone_name} zone.

{state}

{directive}

{user_messages}

Respond with JSON only:
{{"action":"set_temperature"|"no_change","temperature":<{temp_min}-{temp_max} or null>,"reasoning":"<brief>","message_to_user":"<optional or null>"}}"""

# ── Module-level state ────────────────────────────────────────────
_config: dict = {}
_last_evaluation_result: Optional[dict] = None
_evaluation_counter: int = 0  # Incremented each completed cycle
_user_triggered: bool = False  # True when evaluation was triggered by a user message
_telegram_send_fn = None  # Set by telegram_bot on startup


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    global _config
    with open(path) as f:
        _config = yaml.safe_load(f)
    return _config


def get_config() -> dict:
    return _config


def set_telegram_send_fn(fn):
    """Register the Telegram send function (called by telegram_bot)."""
    global _telegram_send_fn
    _telegram_send_fn = fn


async def send_telegram(message: str):
    """Send a message via Telegram if the send function is registered."""
    if _telegram_send_fn:
        try:
            logger.info("Sending Telegram: %s", message[:80])
            await _telegram_send_fn(message)
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
    else:
        logger.warning("Telegram send function not registered — message dropped: %s", message[:80])


def call_llm(system_prompt: str) -> Tuple[Optional[str], bool]:
    """
    Call the local LLM via llama.cpp /v1/chat/completions.
    Returns (response_text, is_valid_json).
    Retries once on JSON failure.
    """
    llm_cfg = _config["llm"]
    payload = {
        "model": llm_cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Make a climate decision based on the context above."}
        ],
        "temperature": llm_cfg["temperature"],
        "top_p": llm_cfg["top_p"],
        "max_tokens": llm_cfg["max_tokens"]
    }

    for attempt in range(2):
        try:
            resp = requests.post(
                llm_cfg["endpoint"],
                json=payload,
                timeout=llm_cfg["timeout_seconds"]
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Try to parse as JSON
            try:
                json.loads(content)
                return content, True
            except json.JSONDecodeError:
                if attempt == 0:
                    logger.warning("LLM returned invalid JSON (attempt 1), retrying...")
                    continue
                return content, False

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out (attempt %d)", attempt + 1)
            if attempt == 0:
                continue
            return "TIMEOUT", False
        except requests.exceptions.ConnectionError:
            logger.error("LLM connection error — is llama-server running on %s?",
                         llm_cfg["endpoint"])
            return "CONNECTION_ERROR", False
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"ERROR: {e}", False

    return None, False


def validate_response(response_text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate LLM response JSON structure.
    Reuses logic from test_qwen_4b.py validate_response().
    """
    try:
        data = json.loads(response_text)

        if "action" not in data:
            return False, "Missing 'action' field"

        if data["action"] not in ["set_temperature", "no_change"]:
            return False, f"Invalid action: {data['action']}"

        if "reasoning" not in data:
            return False, "Missing 'reasoning' field"

        if data["action"] == "set_temperature":
            if "temperature" not in data or data["temperature"] is None:
                return False, "set_temperature action missing temperature"
            if not (TEMP_MIN <= data["temperature"] <= TEMP_MAX):
                return False, f"Temperature {data['temperature']} out of range ({TEMP_MIN}-{TEMP_MAX})"

        if data["action"] == "no_change":
            if "temperature" in data and data["temperature"] is not None:
                return False, "no_change action should not set temperature"

        return True, None

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def build_context(thermo_state, weather_data, all_states=None) -> str:
    """
    Assemble a compact prompt for Qwen 4B.
    Python does the heavy reasoning — determines the situation, comfort range,
    and directive. The LLM just makes the final call with clear, short context.
    """
    recent_msgs = database.get_recent_messages(10)
    now = datetime.now()
    comfort = _config.get("comfort", {})
    sched = _config.get("schedule", {})

    # ── Build state summary (compact) ──
    state_lines = [
        f"Indoor: {thermo_state.indoor_temp}F, {thermo_state.humidity}% humidity",
        f"Mode: {thermo_state.mode}, target: {thermo_state.target_temp or 'none'}F",
        f"Outdoor: {weather_data.current_temp}F, {weather_data.forecast_summary}",
        f"Time: {now.strftime('%I:%M %p')} {now.strftime('%A')}",
    ]
    if all_states:
        for s in all_states:
            if s.device_id != thermo_state.device_id:
                state_lines.append(f"Other zone {s.name}: {s.indoor_temp}F, target={s.target_temp}F")

    # ── Analyze situation in Python and build directive ──
    directive = _build_directive(thermo_state, weather_data, now, comfort, sched, recent_msgs)

    # ── Format user messages (compact) ──
    user_msg_section = _format_user_messages(recent_msgs, now)

    return SYSTEM_PROMPT.format(
        zone_name=thermo_state.name,
        state="\n".join(state_lines),
        directive=directive,
        user_messages=user_msg_section,
        temp_min=TEMP_MIN,
        temp_max=TEMP_MAX,
    )


def _get_message_age_minutes(msg: dict, now: datetime) -> Optional[float]:
    """Get the age of a message in minutes, or None if timestamp is missing.
    Note: SQLite stores timestamps as UTC via datetime('now'), and `now` is
    local time from datetime.now(). We convert the UTC timestamp to local time
    before comparing.
    """
    ts = msg.get("timestamp", "")
    if not ts:
        return None
    try:
        # SQLite datetime('now') returns UTC. Parse as UTC, convert to local.
        from datetime import timezone
        msg_time_utc = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        msg_time_local = msg_time_utc.astimezone().replace(tzinfo=None)
        return (now - msg_time_local).total_seconds() / 60
    except (ValueError, TypeError):
        return None


def _find_recent_user_request(messages: list, now: datetime, max_hours: float) -> Optional[str]:
    """
    Find the most recent user message that looks like a temperature request.
    Returns the message text if it's within max_hours, else None.
    """
    temp_pattern = re.compile(r'(?:set|change|make|turn|put|adjust).*\d+', re.IGNORECASE)
    for msg in reversed(messages):
        text = msg.get("text", "")
        if not text:
            continue
        age = _get_message_age_minutes(msg, now)
        if age is not None and age <= max_hours * 60 and temp_pattern.search(text):
            return text
    return None


def _get_time_period(now: datetime, sched: dict) -> str:
    """Determine the current time period: sleep, pre_wake, waking_up, work, winding_down, home."""
    sleep_time = sched.get("sleep_time", "23:00")
    wake_time = sched.get("wake_time", "07:00")
    sleep_h, sleep_m = map(int, sleep_time.split(":"))
    wake_h, wake_m = map(int, wake_time.split(":"))

    cur = now.hour * 60 + now.minute
    sleep_min = sleep_h * 60 + sleep_m
    wake_min = wake_h * 60 + wake_m
    pre_heat = _config.get("comfort", {}).get("pre_heat_minutes", 30)

    if cur >= sleep_min or cur < wake_min - pre_heat:
        return "sleep"
    if wake_min - pre_heat <= cur < wake_min:
        return "pre_wake"
    if wake_min <= cur < wake_min + 60:
        return "waking_up"
    if 9 * 60 <= cur <= 17 * 60 and now.weekday() < 5:
        return "work"
    if cur >= sleep_min - 60:
        return "winding_down"
    return "home"


def _build_directive(thermo_state, weather_data, now, comfort, sched, messages) -> str:
    """
    Python pre-processor: analyze the situation and produce a clear, concise
    directive for the LLM. This is where the intelligence lives.
    """
    mode = thermo_state.mode  # "cooling", "heating", "auto", "off"
    indoor = thermo_state.indoor_temp
    outdoor = weather_data.current_temp
    target = thermo_state.target_temp
    period = _get_time_period(now, sched)
    user_request_hours = comfort.get("user_request_hours", 2)

    # Comfort ranges
    if mode == "heating":
        comfort_low, comfort_high = comfort.get("winter_range", [68, 72])
    else:
        comfort_low, comfort_high = comfort.get("summer_range", [75, 80])

    parts = [f"Comfort range: {comfort_low}-{comfort_high}F."]

    # ── Check for recent user request (highest priority) ──
    recent_request = _find_recent_user_request(messages, now, user_request_hours)
    if recent_request:
        parts.append(f"PRIORITY: User recently said: '{recent_request}'. Follow this request exactly.")
        return "DIRECTIVE: " + " ".join(parts)

    # Check for expired user request
    expired_request = _find_recent_user_request(messages, now, 24)  # Look back 24h
    if expired_request and not recent_request:
        parts.append(f"User previously said '{expired_request}' but that was over {user_request_hours}h ago. Re-evaluate freely.")

    # ── Sleep time logic (handled in Python, not by LLM) ──
    if period == "sleep":
        if mode == "cooling":
            sleep_cool = comfort.get("sleep_cool_temp", 75)
            override_temp = comfort.get("sleep_cool_override_temp", 80)
            if indoor <= sleep_cool + 1:
                parts.append(f"Sleep time, summer. House is cool ({indoor}F). Let it coast — prefer no_change.")
            elif outdoor > override_temp:
                parts.append(f"Sleep time, summer. Outdoor is hot ({outdoor}F). Indoor {indoor}F is warm. Cool to {sleep_cool}F.")
            else:
                parts.append(f"Sleep time, summer. Outdoor is mild ({outdoor}F). Let the house coast — prefer no_change.")
        else:  # HEAT
            if indoor >= TEMP_MIN + 2:
                parts.append(f"Sleep time, winter. Indoor {indoor}F is fine. Let it drift — prefer no_change.")
            else:
                parts.append(f"Sleep time, winter. Indoor {indoor}F is getting cold. Heat to {comfort_low}F.")

    elif period == "pre_wake":
        if mode == "heating":
            parts.append(f"Early morning, {comfort.get('pre_heat_minutes', 30)} min before wake. Pre-heat to {comfort_high}F so the house is warm.")
        else:
            parts.append(f"Early morning before wake. Check if comfortable, adjust if needed.")

    elif period == "winding_down":
        if mode == "cooling":
            sleep_cool = comfort.get("sleep_cool_temp", 75)
            parts.append(f"Approaching bedtime. Pre-cool to {sleep_cool}F for a comfortable night.")
        else:
            parts.append(f"Approaching bedtime. Let the temp settle toward {comfort_low}F.")

    elif period == "work":
        parts.append(f"Work hours, user may be away. Wider range OK to save energy.")

    elif period == "waking_up":
        parts.append(f"Just woke up. Target {comfort_high}F for comfort.")

    else:  # home
        if indoor < comfort_low:
            parts.append(f"Indoor {indoor}F is below comfort range. Adjust to {comfort_low}F.")
        elif indoor > comfort_high:
            parts.append(f"Indoor {indoor}F is above comfort range. Adjust to {comfort_high}F.")
        else:
            parts.append(f"Indoor {indoor}F is within comfort range. No change likely needed.")

    # ── Forecast analysis (Python-parsed, time-aware) ──
    forecast = weather.get_forecast_analysis()
    if forecast:
        if forecast.pre_cool_now and mode == "cooling":
            parts.append(f"URGENT: {forecast.advisory} Pre-cool to {comfort_low}F now.")
        elif forecast.pre_heat_now and mode == "heating":
            parts.append(f"URGENT: {forecast.advisory} Pre-heat to {comfort_high}F now.")
        elif forecast.advisory:
            parts.append(f"Forecast: {forecast.advisory}")

    if not forecast or not forecast.advisory:
        if outdoor > 100:
            parts.append(f"Extreme heat outside ({outdoor}F). Pre-condition aggressively.")
        elif outdoor < 35:
            parts.append(f"Very cold outside ({outdoor}F). Keep the house warm.")

    # ── Question detection ──
    if messages:
        last_text = messages[-1].get("text", "")
        last_age = _get_message_age_minutes(messages[-1], now)
        if last_age is not None and last_age < 30:
            # Check if it's a question (not a command)
            if "?" in last_text or any(w in last_text.lower() for w in ["what", "how", "when", "is it", "will it"]):
                parts.append(f"User asked: '{last_text}'. Answer in message_to_user. Use no_change.")

    return "DIRECTIVE: " + " ".join(parts)


def _format_user_messages(messages: list, now: datetime) -> str:
    """Format recent user messages — compact, only if relevant."""
    if not messages:
        return ""

    formatted = []
    for msg in messages[-3:]:  # Last 3 only — save tokens
        text = msg.get("text", "")
        if not text:
            continue
        age = _get_message_age_minutes(msg, now)
        if age is not None:
            if age < 60:
                age_str = f"{int(age)}m ago"
            else:
                age_str = f"{age / 60:.1f}h ago"
            formatted.append(f"User ({age_str}): {text}")
        else:
            formatted.append(f"User: {text}")

    if not formatted:
        return ""
    return "Messages:\n" + "\n".join(formatted)


def check_guardrails(decision: dict, user_triggered: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Enforce safety rules before executing a decision.
    Returns (allowed, reason_if_blocked).
    User-triggered evaluations skip the rate limit (user explicitly asked).
    """
    if decision["action"] != "set_temperature":
        return True, None

    temp = decision.get("temperature")

    # Temperature bounds
    if temp is not None and not (TEMP_MIN <= temp <= TEMP_MAX):
        return False, f"Temperature {temp} outside bounds ({TEMP_MIN}-{TEMP_MAX})"

    # Rate limit (skipped for user-triggered evaluations)
    if not user_triggered:
        changes = database.count_temp_changes_last_hour()
        if changes >= MAX_CHANGES_PER_HOUR:
            return False, f"Rate limit: {changes} changes in last hour (max {MAX_CHANGES_PER_HOUR})"

    # Manual override backoff (skipped for user-triggered evaluations)
    if not user_triggered:
        last_override = database.get_last_manual_override_time()
        if last_override:
            elapsed = (datetime.utcnow() - last_override).total_seconds() / 60
            if elapsed < MANUAL_OVERRIDE_BACKOFF_MINUTES:
                remaining = int(MANUAL_OVERRIDE_BACKOFF_MINUTES - elapsed)
                return False, f"Manual override backoff: {remaining} minutes remaining"

    return True, None


async def execute_decision(decision: dict, thermo_state, weather_data, raw_response: str):
    """Execute the agent's decision — call Nest API if needed, log to DB."""
    reasoning = decision.get("reasoning", "")
    temperature = decision.get("temperature")
    zone = thermo_state.name

    # Skip redundant temperature changes BEFORE logging
    if decision["action"] == "set_temperature" and temperature is not None:
        current_target = thermo_state.target_temp
        if current_target is not None and abs(temperature - current_target) < 0.5:
            logger.info("[%s] Skipping — target already at %.0fF", zone, current_target)
            decision["action"] = "no_change"
            decision["reasoning"] = f"Already at {current_target:.0f}F (requested {temperature:.0f}F)"

    action = decision["action"]

    # Log the decision (now reflects the actual action taken)
    database.log_decision(
        indoor_temp=thermo_state.indoor_temp,
        outdoor_temp=weather_data.current_temp,
        action=action,
        temperature=temperature if action == "set_temperature" else None,
        reasoning=reasoning,
        raw_response=raw_response,
        zone=zone
    )

    # Log climate snapshot
    database.log_climate(
        indoor_temp=thermo_state.indoor_temp,
        indoor_humidity=thermo_state.humidity,
        outdoor_temp=weather_data.current_temp,
        outdoor_humidity=weather_data.humidity,
        forecast=weather_data.forecast_summary,
        hvac_mode=thermo_state.mode,
        hvac_running=thermo_state.hvac_running,
        target_temp=thermo_state.target_temp or 0,
        action=action,
        reasoning=reasoning,
        zone=zone
    )

    # Execute temperature change
    if action == "set_temperature" and temperature is not None:
        success = await asyncio.to_thread(
            nest_api.set_temperature, temperature, thermo_state.device_id
        )
        if success:
            logger.info("[%s] Temperature set to %.0fF (was %.0fF) — %s",
                        zone, temperature, thermo_state.target_temp or 0, reasoning)
        else:
            logger.error("[%s] Failed to set temperature to %.0fF", zone, temperature)
            database.log_error("nest_api", "set_temperature_failed",
                               f"[{zone}] Tried to set {temperature}F")

    return action


def generate_weekly_report() -> str:
    """Generate a Sunday summary from climate_log."""
    since = datetime.utcnow() - timedelta(days=7)
    entries = database.get_climate_log_since(since)

    if not entries:
        return "No climate data recorded this week."

    temps = [e["indoor_temp"] for e in entries if e["indoor_temp"]]
    outdoor = [e["outdoor_temp"] for e in entries if e["outdoor_temp"]]
    changes = [e for e in entries if e["action"] == "set_temperature"]
    running = [e for e in entries if e["hvac_running"]]

    report = "Weekly Climate Report\n"
    report += "=" * 30 + "\n"
    if temps:
        report += f"Indoor: avg {sum(temps)/len(temps):.1f}F, "
        report += f"range {min(temps):.0f}-{max(temps):.0f}F\n"
    if outdoor:
        report += f"Outdoor: avg {sum(outdoor)/len(outdoor):.1f}F, "
        report += f"range {min(outdoor):.0f}-{max(outdoor):.0f}F\n"
    report += f"Temperature changes: {len(changes)}\n"
    report += f"HVAC active snapshots: {len(running)} of {len(entries)}\n"
    report += f"Total data points: {len(entries)}\n"

    return report


async def run_evaluation_cycle() -> Optional[dict]:
    """
    Run one complete evaluation cycle for ALL thermostats.
    Each zone gets its own LLM call with awareness of other zones.
    Returns the last decision (for Telegram response).
    """
    global _user_triggered
    user_triggered = _user_triggered
    _user_triggered = False  # Reset for next cycle
    logger.info("Starting evaluation cycle (user_triggered=%s)", user_triggered)
    last_decision = None
    _cycle_decisions = []  # Track all zone decisions for summary

    try:
        # 1. Get ALL thermostat states
        all_states = await asyncio.to_thread(nest_api.get_all_thermostat_states)
        for s in all_states:
            logger.info("[%s] %.1fF, %d%% humidity, mode=%s, target=%s",
                         s.name, s.indoor_temp, s.humidity, s.mode, s.target_temp)

        # Check for manual overrides on each device
        for s in all_states:
            if nest_api.detect_manual_override(s.device_id, s.target_temp):
                database.log_manual_override(
                    s.target_temp,
                    nest_api._last_known_targets.get(s.device_id, 0),
                    zone=s.name
                )
                logger.info("[%s] Manual override detected", s.name)
                await send_telegram(
                    f"[{s.name}] Manual override detected "
                    f"(target changed to {s.target_temp}F). "
                    f"Backing off for {MANUAL_OVERRIDE_BACKOFF_MINUTES} minutes."
                )

        # 2. Get weather (shared across zones)
        weather_data = await asyncio.to_thread(weather.get_weather)
        logger.info("Weather: %.1fF, %d%% humidity, stale=%s",
                     weather_data.current_temp, weather_data.humidity, weather_data.is_stale)

        # Check weather alerts (once)
        alerts = weather.check_weather_alerts(weather_data)
        for alert in alerts:
            logger.warning("Weather alert: %s", alert)
            await send_telegram(f"Weather Alert: {alert}")

        # 3. Start LLM server
        server_started = await asyncio.to_thread(llm_server.start)
        if not server_started:
            logger.error("Failed to start LLM server — skipping evaluation")
            database.log_error("llm_server", "start_failed", "Could not start llama-server")
            return None

        # 4-7. Evaluate each zone
        for thermo_state in all_states:
            zone = thermo_state.name
            logger.info("[%s] Running LLM evaluation...", zone)

            # Build context with awareness of all zones
            system_prompt = build_context(thermo_state, weather_data, all_states)
            response_text, is_json = await asyncio.to_thread(call_llm, system_prompt)

            if not is_json:
                logger.error("[%s] LLM invalid JSON: %s", zone, response_text[:200])
                database.log_error("llm", "invalid_json", f"[{zone}] {response_text[:500]}")
                continue

            is_valid, error_msg = validate_response(response_text)
            if not is_valid:
                logger.error("[%s] Validation failed: %s", zone, error_msg)
                database.log_error("llm", "validation_failed", f"[{zone}] {error_msg}")
                continue

            decision = json.loads(response_text)

            # Check guardrails
            allowed, block_reason = check_guardrails(decision, user_triggered=user_triggered)
            if not allowed:
                logger.warning("[%s] Guardrail blocked: %s", zone, block_reason)
                database.log_decision(
                    indoor_temp=thermo_state.indoor_temp,
                    outdoor_temp=weather_data.current_temp,
                    action=f"BLOCKED:{decision['action']}",
                    temperature=decision.get("temperature"),
                    reasoning=f"BLOCKED: {block_reason} | Original: {decision.get('reasoning', '')}",
                    raw_response=response_text,
                    zone=zone
                )
                continue

            # Execute
            await execute_decision(decision, thermo_state, weather_data, response_text)
            decision["zone"] = zone
            _cycle_decisions.append(decision)
            last_decision = decision

            logger.info("[%s] Cycle complete: action=%s, temp=%s",
                         zone, decision["action"], decision.get("temperature"))

        # 8. Stop LLM server to free GPU
        await asyncio.to_thread(llm_server.stop)

        # 9. Update heartbeat
        database.update_heartbeat()

        # 9. Send Telegram notifications
        # Temperature changes
        temp_changes = [d for d in _cycle_decisions if d["action"] == "set_temperature"]
        if temp_changes:
            parts = []
            for d in temp_changes:
                reason = d.get("reasoning", "")[:60]
                parts.append(f"{d['zone']}: set to {d['temperature']:.0f}F — {reason}")
            await send_telegram("\n".join(parts))

        # LLM message_to_user is only delivered for user-triggered cycles
        # (handled by telegram_bot.handle_message). Don't send unsolicited
        # LLM commentary during scheduled 20-min cycles.

        return last_decision

    except Exception as e:
        logger.error("Evaluation cycle failed: %s", e, exc_info=True)
        database.log_error("agent", type(e).__name__, str(e))
        await asyncio.to_thread(llm_server.stop)
        return None


def trigger_evaluation():
    """Called by Telegram bot to wake the agent loop for an immediate evaluation."""
    global _user_triggered
    _user_triggered = True
    logger.info("Evaluation triggered by user message")


async def agent_loop():
    """
    Main async loop — runs evaluation every N minutes or when triggered.
    Uses asyncio.Event for event-based interruption.
    """
    global _last_evaluation_result, _evaluation_counter, _user_triggered

    interval_seconds = _config["agent"]["loop_interval_minutes"] * 60

    logger.info("Agent loop starting (interval: %d minutes)", interval_seconds // 60)

    # Wait for Telegram bot to register its send function
    logger.info("Waiting for Telegram bot to initialize...")
    for i in range(15):  # Up to 30 seconds (should be <5s now)
        if _telegram_send_fn is not None:
            logger.info("Telegram bot ready (took %ds)", (i + 1) * 2)
            break
        await asyncio.sleep(2)
    else:
        logger.warning("Telegram bot did not register in time — summary messages will be skipped")

    # Weekly report day check
    last_report_day = None

    # Daily cleanup
    last_cleanup_day = None

    while True:
        try:
            # Run evaluation
            result = await asyncio.wait_for(
                run_evaluation_cycle(),
                timeout=300  # 5-minute max per cycle
            )
            _last_evaluation_result = result
            _evaluation_counter += 1

            # Weekly report on Sundays
            now = datetime.now()
            if now.weekday() == 6 and last_report_day != now.date():
                report = generate_weekly_report()
                await send_telegram(report)
                last_report_day = now.date()

            # Daily cleanup
            if last_cleanup_day != now.date():
                database.cleanup_old_records()
                last_cleanup_day = now.date()

        except asyncio.TimeoutError:
            logger.error("Evaluation cycle timed out after 5 minutes")
            database.log_error("agent", "timeout", "Evaluation cycle exceeded 5 minutes")
        except Exception as e:
            logger.error("Agent loop error: %s", e, exc_info=True)
            database.log_error("agent", type(e).__name__, str(e))

        # If a trigger came in during the cycle, run again immediately
        if _user_triggered:
            logger.info("Trigger received during cycle — running again immediately")
            continue

        # Wait for next interval, checking for triggers every 2 seconds
        waited = 0
        while waited < interval_seconds:
            await asyncio.sleep(2)
            waited += 2
            if _user_triggered:
                logger.info("Agent loop woken by user trigger (waited %ds)", waited)
                break


def get_last_evaluation_result() -> Optional[dict]:
    """Get the result of the last evaluation cycle (for /status command)."""
    return _last_evaluation_result


def get_evaluation_counter() -> int:
    """Get the evaluation counter (for detecting new cycles)."""
    return _evaluation_counter


async def main():
    """Entry point — init all modules, run agent loop + Telegram bot."""
    # Load config
    config = load_config()

    # Setup logging
    log_level = config["agent"].get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger.info("AI Thermostat Agent starting...")

    # Init database
    database.init_db(config["agent"]["db_path"])

    # Init weather
    w = config["weather"]
    weather.init_weather(w["api_key"], w["latitude"], w["longitude"],
                         w.get("cache_minutes", 20), w.get("stale_hours", 6))

    # Init Nest API
    nest_api.init_nest(config["nest"]["tokens_path"],
                       devices=config["nest"].get("devices"))

    # Init LLM server manager (doesn't start the server yet)
    llm_cfg = config["llm"]
    llm_port = int(llm_cfg["endpoint"].split(":")[-1].split("/")[0])
    llm_server.init(
        server_exe=llm_cfg["server_exe"],
        model_path=llm_cfg["model_path"],
        port=llm_port
    )

    # Init Telegram bot (imported here to avoid circular imports)
    import telegram_bot
    telegram_bot.init_bot(config)

    logger.info("All modules initialized. Starting agent loop + Telegram bot.")

    # Run agent loop and Telegram bot concurrently
    await asyncio.gather(
        agent_loop(),
        telegram_bot.start_bot()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user.")
        sys.exit(0)
