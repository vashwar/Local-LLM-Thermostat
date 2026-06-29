#!/usr/bin/env python3
"""
NLP Parser — parses user messages into structured commands.

Uses LLM for parsing only (not decision-making).
Falls back to regex when LLM is unavailable.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedCommand:
    action_type: str              # "set_temp", "adjust", "status_query", "question", "unknown"
    target_temp: Optional[float] = None   # Explicit temperature, or None
    zone: Optional[str] = None    # "upstairs", "downstairs", "both", or None
    direction: Optional[str] = None  # "warmer", "cooler", or None
    raw_text: str = ""            # Original message


# ── Zone aliases ─────────────────────────────────────────────────

ZONE_ALIASES = {
    "upstairs": "upstairs",
    "bedroom": "upstairs",
    "bed": "upstairs",
    "up": "upstairs",
    "downstairs": "downstairs",
    "kitchen": "downstairs",
    "down": "downstairs",
    "both": "both",
    "all": "both",
    "house": "both",
    "everything": "both",
    "everywhere": "both",
}


# ── Regex fallback parser ────────────────────────────────────────

def _parse_regex(text: str) -> ParsedCommand:
    """Parse user message using regex patterns. Handles common patterns."""
    original = text
    text = text.lower().strip()

    # "set [zone] to <temp>" or "set <temp> [zone]"
    m = re.match(r"set\s+(?:(?:the\s+)?(\w+)\s+)?to\s+(\d{2})", text)
    if m:
        zone_word = m.group(1)
        temp = float(m.group(2))
        zone = ZONE_ALIASES.get(zone_word) if zone_word else None
        return ParsedCommand(action_type="set_temp", target_temp=temp,
                           zone=zone, raw_text=original)

    # "<temp> [zone]" or "[zone] <temp>"
    m = re.match(r"(\d{2})\s*(?:degrees?\s*)?(?:for\s+)?(\w+)?$", text)
    if m:
        temp = float(m.group(1))
        zone_word = m.group(2)
        zone = ZONE_ALIASES.get(zone_word) if zone_word else None
        if 60 <= temp <= 85:
            return ParsedCommand(action_type="set_temp", target_temp=temp,
                               zone=zone, raw_text=original)

    # "make it warmer/cooler" or "turn it up/down"
    if any(w in text for w in ["warmer", "warm it", "turn.*up", "too cold", "i'm cold",
                                 "im cold", "freezing"]):
        return ParsedCommand(action_type="adjust", direction="warmer", raw_text=original)

    if any(w in text for w in ["cooler", "cool it", "turn.*down", "too hot", "i'm hot",
                                 "im hot", "i am hot", "so hot"]):
        return ParsedCommand(action_type="adjust", direction="cooler", raw_text=original)

    # Questions
    if "?" in text or any(text.startswith(w) for w in
                          ["what", "how", "when", "why", "is it", "will it",
                           "can you", "could you", "do you"]):
        return ParsedCommand(action_type="question", raw_text=original)

    # Status queries
    if any(w in text for w in ["status", "temperature", "temp", "current"]):
        return ParsedCommand(action_type="status_query", raw_text=original)

    return ParsedCommand(action_type="unknown", raw_text=original)


# ── LLM parser ──────────────────────────────────────────────────

NLP_SYSTEM_PROMPT = """Parse the user message into a JSON command. Output ONLY JSON.

Examples:
"set upstairs to 77" → {"action_type":"set_temp","target_temp":77,"zone":"upstairs","direction":null}
"make it cooler" → {"action_type":"adjust","target_temp":null,"zone":null,"direction":"cooler"}
"I'm hot" → {"action_type":"adjust","target_temp":null,"zone":null,"direction":"cooler"}
"set both to 79" → {"action_type":"set_temp","target_temp":79,"zone":"both","direction":null}
"what's the temperature?" → {"action_type":"question","target_temp":null,"zone":null,"direction":null}
"set to 76" → {"action_type":"set_temp","target_temp":76,"zone":null,"direction":null}
"77 downstairs" → {"action_type":"set_temp","target_temp":77,"zone":"downstairs","direction":null}

Zone aliases: bedroom/bed=upstairs, kitchen=downstairs, both/all/house=both

Parse this message:"""


def _parse_llm_response(response_text: str, original_text: str) -> Optional[ParsedCommand]:
    """Parse LLM JSON response into ParsedCommand. Returns None on failure."""
    try:
        # Strip markdown fences
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        data = json.loads(text)
        action_type = data.get("action_type", "unknown")
        if action_type not in ("set_temp", "adjust", "status_query", "question", "unknown"):
            return None

        return ParsedCommand(
            action_type=action_type,
            target_temp=data.get("target_temp"),
            zone=data.get("zone"),
            direction=data.get("direction"),
            raw_text=original_text,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def parse(text: str, call_llm_fn=None) -> ParsedCommand:
    """
    Parse a user message into a structured command.

    Args:
        text: User's message
        call_llm_fn: Optional function(system_prompt) -> (response_text, is_json)
                     If None or LLM fails, falls back to regex.
    """
    if not text or not text.strip():
        return ParsedCommand(action_type="unknown", raw_text=text or "")

    # Try LLM first if available
    if call_llm_fn:
        try:
            prompt = NLP_SYSTEM_PROMPT + f' "{text}"'
            response_text, is_json = call_llm_fn(prompt)
            if is_json and response_text:
                result = _parse_llm_response(response_text, text)
                if result:
                    logger.info("LLM parsed: '%s' → %s", text, result.action_type)
                    return result
            logger.warning("LLM parse failed, falling back to regex")
        except Exception as e:
            logger.warning("LLM parse error: %s, falling back to regex", e)

    # Regex fallback
    result = _parse_regex(text)
    logger.info("Regex parsed: '%s' → %s", text, result.action_type)
    return result
