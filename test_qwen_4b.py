#!/usr/bin/env python3
"""
Qwen 4B JSON Reliability Test Harness
Tests whether Qwen 4B can reliably produce valid JSON for climate decisions.
"""

import json
import requests
import time
from datetime import datetime
from typing import Tuple, Optional

# llama.cpp server endpoint (default)
LLAMA_CPP_URL = "http://localhost:8000/v1/chat/completions"

# System prompt for thermostat decisions
SYSTEM_PROMPT = """You are a smart thermostat agent. Your job is to decide the optimal
temperature setting for a home based on the context provided.

CURRENT STATE:
- Indoor: {temp}F, {humidity}% humidity
- Outdoor: {current_temp}F, forecast: {forecast_summary}
- HVAC mode: {mode}, current target: {target}F
- Schedule: {schedule_context}
- Recent messages: {recent_messages}
- Time: {current_time}, Day: {day_of_week}

Respond ONLY with valid JSON in this exact format:
{{
  "action": "set_temperature" | "no_change",
  "temperature": <number between 60-85, or null if no_change>,
  "reasoning": "<1-2 sentence explanation>",
  "message_to_user": "<optional message to send via Telegram, or null>"
}}

NOTE: "action" determines the thermostat action only. "message_to_user"
is independent — the agent can send a Telegram message alongside ANY
action."""


# Test scenarios covering various climate conditions
TEST_SCENARIOS = [
    {
        "name": "Normal day (70F indoor, 85F outdoor)",
        "context": {
            "temp": 70,
            "humidity": 45,
            "current_temp": 85,
            "forecast_summary": "sunny, 88F by afternoon",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "weekday, user at work 9-5",
            "recent_messages": "none",
            "current_time": "10:00 AM",
            "day_of_week": "Monday"
        }
    },
    {
        "name": "Heat wave (76F indoor, 102F outdoor)",
        "context": {
            "temp": 76,
            "humidity": 55,
            "current_temp": 102,
            "forecast_summary": "heat advisory, 105F peak 1-5pm",
            "mode": "cooling",
            "target": 74,
            "schedule_context": "weekday, user at work",
            "recent_messages": "none",
            "current_time": "11:00 AM",
            "day_of_week": "Wednesday"
        }
    },
    {
        "name": "Morning wake-up (68F, user waking up)",
        "context": {
            "temp": 68,
            "humidity": 40,
            "current_temp": 72,
            "forecast_summary": "mild morning, warming to 82F",
            "mode": "cooling",
            "target": 68,
            "schedule_context": "weekend, user waking up at 6:30am",
            "recent_messages": "none",
            "current_time": "6:30 AM",
            "day_of_week": "Saturday"
        }
    },
    {
        "name": "User message: heading out",
        "context": {
            "temp": 72,
            "humidity": 42,
            "current_temp": 88,
            "forecast_summary": "sunny, 92F by 2pm",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "weekday, user usually at work 9-5",
            "recent_messages": "User: 'We are heading out for a few hours'",
            "current_time": "2:00 PM",
            "day_of_week": "Tuesday"
        }
    },
    {
        "name": "User message: set temperature",
        "context": {
            "temp": 74,
            "humidity": 48,
            "current_temp": 92,
            "forecast_summary": "hot, 95F",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "home",
            "recent_messages": "Wife: 'It's too cold, can you set it to 74'",
            "current_time": "3:30 PM",
            "day_of_week": "Thursday"
        }
    },
    {
        "name": "Pre-cooling before heat spike",
        "context": {
            "temp": 71,
            "humidity": 40,
            "current_temp": 82,
            "forecast_summary": "100F peak expected 2-5pm",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "user home all day",
            "recent_messages": "none",
            "current_time": "10:00 AM",
            "day_of_week": "Friday"
        }
    },
    {
        "name": "Stale weather data (>6 hrs old)",
        "context": {
            "temp": 75,
            "humidity": 50,
            "current_temp": 85,
            "forecast_summary": "[STALE - last update 8 hours ago] 90F",
            "mode": "cooling",
            "target": 74,
            "schedule_context": "home",
            "recent_messages": "none",
            "current_time": "4:00 PM",
            "day_of_week": "Monday"
        }
    },
    {
        "name": "Edge case: temp boundary (85F)",
        "context": {
            "temp": 85,
            "humidity": 60,
            "current_temp": 105,
            "forecast_summary": "extreme heat, 108F",
            "mode": "cooling",
            "target": 75,
            "schedule_context": "emergency heat event",
            "recent_messages": "none",
            "current_time": "2:00 PM",
            "day_of_week": "Sunday"
        }
    },
    {
        "name": "Edge case: temp boundary (60F)",
        "context": {
            "temp": 60,
            "humidity": 30,
            "current_temp": 32,
            "forecast_summary": "freezing, 28F by evening",
            "mode": "heating",
            "target": 68,
            "schedule_context": "cold snap alert",
            "recent_messages": "none",
            "current_time": "6:00 PM",
            "day_of_week": "Tuesday"
        }
    },
    {
        "name": "Evening (6 PM, winding down)",
        "context": {
            "temp": 72,
            "humidity": 45,
            "current_temp": 78,
            "forecast_summary": "cooling to 72F overnight",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "user arriving home soon",
            "recent_messages": "none",
            "current_time": "6:00 PM",
            "day_of_week": "Wednesday"
        }
    },
    {
        "name": "Night (11 PM, sleep schedule active)",
        "context": {
            "temp": 68,
            "humidity": 50,
            "current_temp": 65,
            "forecast_summary": "cool night, 62F",
            "mode": "off",
            "target": 68,
            "schedule_context": "sleep time (10pm-6:30am), prefer cooler",
            "recent_messages": "none",
            "current_time": "11:00 PM",
            "day_of_week": "Thursday"
        }
    },
    {
        "name": "Rapid schedule change (user home unexpectedly)",
        "context": {
            "temp": 76,
            "humidity": 48,
            "current_temp": 94,
            "forecast_summary": "warm afternoon, 96F",
            "mode": "cooling",
            "target": 78,
            "schedule_context": "expected away, but user messaging they're home",
            "recent_messages": "User: 'Got home early, it's a bit warm'",
            "current_time": "1:00 PM",
            "day_of_week": "Friday"
        }
    },
    {
        "name": "Multiple messages in one cycle",
        "context": {
            "temp": 73,
            "humidity": 46,
            "current_temp": 89,
            "forecast_summary": "hot, 91F",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "home",
            "recent_messages": "User: 'Heading out' -> Wife: 'Make sure it's cool when we get back' -> User: '6pm return'",
            "current_time": "2:30 PM",
            "day_of_week": "Saturday"
        }
    },
    {
        "name": "Humidity extremes (very dry)",
        "context": {
            "temp": 72,
            "humidity": 15,
            "current_temp": 95,
            "forecast_summary": "dry, low humidity continues",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "home",
            "recent_messages": "none",
            "current_time": "12:00 PM",
            "day_of_week": "Monday"
        }
    },
    {
        "name": "Humidity extremes (very humid)",
        "context": {
            "temp": 75,
            "humidity": 80,
            "current_temp": 85,
            "forecast_summary": "humid, 85F with rain expected",
            "mode": "cooling",
            "target": 73,
            "schedule_context": "home",
            "recent_messages": "none",
            "current_time": "3:00 PM",
            "day_of_week": "Tuesday"
        }
    },
    {
        "name": "Forecast just changed (new data)",
        "context": {
            "temp": 71,
            "humidity": 42,
            "current_temp": 84,
            "forecast_summary": "UPDATED: heat advisory issued, 103F by 4pm (was 88F)",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "home",
            "recent_messages": "none",
            "current_time": "11:30 AM",
            "day_of_week": "Wednesday"
        }
    },
    {
        "name": "Edge: no target set (first run)",
        "context": {
            "temp": 70,
            "humidity": 45,
            "current_temp": 80,
            "forecast_summary": "clear, 85F",
            "mode": "off",
            "target": "none (first run)",
            "schedule_context": "initial setup",
            "recent_messages": "none",
            "current_time": "9:00 AM",
            "day_of_week": "Monday"
        }
    },
    {
        "name": "Conflicting user preferences (user vs wife)",
        "context": {
            "temp": 72,
            "humidity": 45,
            "current_temp": 88,
            "forecast_summary": "warm, 90F",
            "mode": "cooling",
            "target": 72,
            "schedule_context": "home",
            "recent_messages": "User: 'It's too warm' -> Wife: 'It's too cold now, I want 74'",
            "current_time": "2:00 PM",
            "day_of_week": "Thursday"
        }
    },
]


def call_llama_cpp(context: dict) -> Tuple[Optional[str], bool]:
    """
    Call llama.cpp server with the climate context.
    Returns (response_text, is_valid_json)
    """
    system_prompt = SYSTEM_PROMPT.format(**context)

    payload = {
        "model": "qwen4b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Make a climate decision based on the context above."}
        ],
        "temperature": 0.3,  # Low temperature for deterministic output
        "top_p": 0.9,
        "max_tokens": 500
    }

    try:
        response = requests.post(LLAMA_CPP_URL, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Try to parse as JSON
        try:
            json.loads(content)
            return content, True
        except json.JSONDecodeError:
            return content, False

    except requests.exceptions.Timeout:
        return "TIMEOUT", False
    except requests.exceptions.ConnectionError:
        return "CONNECTION_ERROR", False
    except Exception as e:
        return f"ERROR: {str(e)}", False


def validate_response(response_text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that the response is valid JSON with correct structure.
    Returns (is_valid, error_message)
    """
    try:
        data = json.loads(response_text)

        # Check required fields
        if "action" not in data:
            return False, "Missing 'action' field"

        if data["action"] not in ["set_temperature", "no_change"]:
            return False, f"Invalid action: {data['action']}"

        if "reasoning" not in data:
            return False, "Missing 'reasoning' field"

        # Validate temperature field
        if data["action"] == "set_temperature":
            if "temperature" not in data or data["temperature"] is None:
                return False, "set_temperature action missing temperature"
            if not (60 <= data["temperature"] <= 85):
                return False, f"Temperature {data['temperature']} out of range (60-85)"

        if data["action"] == "no_change":
            if "temperature" in data and data["temperature"] is not None:
                return False, "no_change action should not set temperature"

        return True, None

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"


def run_test_harness():
    """
    Run the Qwen 4B test harness.
    """
    print("=" * 70)
    print("QWEN 4B JSON RELIABILITY TEST HARNESS")
    print("=" * 70)
    print(f"\nTarget: {LLAMA_CPP_URL}")
    print(f"Scenarios: {len(TEST_SCENARIOS)}")
    print(f"Start time: {datetime.now().isoformat()}\n")

    results = {
        "total": 0,
        "success": 0,
        "json_parse_fail": 0,
        "validation_fail": 0,
        "error": 0,
        "failures": []
    }

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n[{i}/{len(TEST_SCENARIOS)}] {scenario['name']}")
        print("-" * 70)

        results["total"] += 1

        # Call LLM
        response_text, is_valid_json = call_llama_cpp(scenario["context"])

        if not is_valid_json:
            print(f"[FAIL] JSON PARSE FAILED")
            print(f"   Response: {response_text[:100]}...")
            results["json_parse_fail"] += 1
            results["failures"].append({
                "scenario": scenario["name"],
                "reason": "JSON parse failed",
                "response": response_text[:200]
            })
            continue

        # Validate structure
        is_valid, error_msg = validate_response(response_text)

        if not is_valid:
            print(f"[FAIL] VALIDATION FAILED: {error_msg}")
            print(f"   Response: {response_text}")
            results["validation_fail"] += 1
            results["failures"].append({
                "scenario": scenario["name"],
                "reason": error_msg,
                "response": response_text
            })
            continue

        # Success
        print(f"[PASS] PASSED")
        data = json.loads(response_text)
        print(f"   Action: {data['action']}")
        if data.get('temperature'):
            print(f"   Temperature: {data['temperature']}F")
        print(f"   Reasoning: {data['reasoning'][:60]}...")
        results["success"] += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total scenarios: {results['total']}")
    print(f"Successful: {results['success']} ({results['success']*100//results['total']}%)")
    print(f"JSON parse failures: {results['json_parse_fail']}")
    print(f"Validation failures: {results['validation_fail']}")
    print(f"Other errors: {results['error']}")

    success_rate = (results['success'] / results['total']) * 100
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")

    # Decision
    # Failures detail (print before decision gate)
    if results["failures"]:
        print("\n" + "=" * 70)
        print("FAILURE DETAILS")
        print("=" * 70)
        for failure in results["failures"]:
            print(f"\nScenario: {failure['scenario']}")
            print(f"Reason: {failure['reason']}")
            print(f"Response: {failure['response'][:150]}...")

    print("\n" + "=" * 70)
    print("DECISION GATE")
    print("=" * 70)

    if success_rate >= 95:
        print("[PASS] GATE PASSED (>95%)")
        print("  -> Proceed to full implementation with Qwen 4B")
        return True
    elif success_rate >= 90:
        print("[WARN] GATE CONDITIONAL (90-94%)")
        print("  -> Try prompt engineering improvements before proceeding")
        print("  -> Current prompt may be too loose; try stricter constraints")
        return None
    else:
        print("[FAIL] GATE FAILED (<90%)")
        print("  -> SWITCH to Qwen 7B or Mistral 7B before building")
        print("  -> Qwen 4B is not reliable enough for this use case")
        return False


if __name__ == "__main__":
    try:
        run_test_harness()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
