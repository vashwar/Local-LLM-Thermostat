"""
Tests for nlp_parser.py — regex fallback and LLM parsing.
"""

import json
import pytest

from nlp_parser import parse, ParsedCommand, _parse_regex, _parse_llm_response


# ── Regex Fallback Tests ─────────────────────────────────────────

class TestRegexFallback:
    def test_set_to_temp(self):
        result = _parse_regex("set to 77")
        assert result.action_type == "set_temp"
        assert result.target_temp == 77
        assert result.zone is None

    def test_set_zone_to_temp(self):
        result = _parse_regex("set upstairs to 77")
        assert result.action_type == "set_temp"
        assert result.target_temp == 77
        assert result.zone == "upstairs"

    def test_set_bedroom_to_temp(self):
        result = _parse_regex("set bedroom to 75")
        assert result.action_type == "set_temp"
        assert result.target_temp == 75
        assert result.zone == "upstairs"  # bedroom → upstairs

    def test_set_kitchen_to_temp(self):
        result = _parse_regex("set kitchen to 76")
        assert result.action_type == "set_temp"
        assert result.target_temp == 76
        assert result.zone == "downstairs"

    def test_set_both_to_temp(self):
        result = _parse_regex("set both to 79")
        assert result.action_type == "set_temp"
        assert result.target_temp == 79
        assert result.zone == "both"

    def test_set_downstairs_to_temp(self):
        result = _parse_regex("set downstairs to 74")
        assert result.action_type == "set_temp"
        assert result.target_temp == 74
        assert result.zone == "downstairs"

    def test_bare_temp(self):
        result = _parse_regex("77")
        assert result.action_type == "set_temp"
        assert result.target_temp == 77

    def test_temp_with_zone(self):
        result = _parse_regex("77 downstairs")
        assert result.action_type == "set_temp"
        assert result.target_temp == 77
        assert result.zone == "downstairs"

    def test_make_it_warmer(self):
        result = _parse_regex("make it warmer")
        assert result.action_type == "adjust"
        assert result.direction == "warmer"

    def test_make_it_cooler(self):
        result = _parse_regex("make it cooler")
        assert result.action_type == "adjust"
        assert result.direction == "cooler"

    def test_im_hot(self):
        result = _parse_regex("I'm hot")
        assert result.action_type == "adjust"
        assert result.direction == "cooler"

    def test_too_cold(self):
        result = _parse_regex("too cold")
        assert result.action_type == "adjust"
        assert result.direction == "warmer"

    def test_im_cold(self):
        result = _parse_regex("I'm cold")
        assert result.action_type == "adjust"
        assert result.direction == "warmer"

    def test_freezing(self):
        result = _parse_regex("it's freezing in here")
        assert result.action_type == "adjust"
        assert result.direction == "warmer"

    def test_so_hot(self):
        result = _parse_regex("it's so hot")
        assert result.action_type == "adjust"
        assert result.direction == "cooler"

    def test_question_mark(self):
        result = _parse_regex("what's the temperature?")
        assert result.action_type == "question"

    def test_question_word(self):
        result = _parse_regex("how warm is it")
        assert result.action_type == "question"

    def test_status_query(self):
        result = _parse_regex("current temperature")
        assert result.action_type == "status_query"

    def test_unknown(self):
        result = _parse_regex("hello there")
        assert result.action_type == "unknown"

    def test_case_insensitive(self):
        result = _parse_regex("SET UPSTAIRS TO 77")
        assert result.action_type == "set_temp"
        assert result.target_temp == 77
        assert result.zone == "upstairs"


# ── LLM Response Parsing ────────────────────────────────────────

class TestLLMResponseParsing:
    def test_valid_set_temp(self):
        response = json.dumps({
            "action_type": "set_temp",
            "target_temp": 77,
            "zone": "upstairs",
            "direction": None,
        })
        result = _parse_llm_response(response, "set upstairs to 77")
        assert result is not None
        assert result.action_type == "set_temp"
        assert result.target_temp == 77
        assert result.zone == "upstairs"

    def test_valid_adjust(self):
        response = json.dumps({
            "action_type": "adjust",
            "target_temp": None,
            "zone": None,
            "direction": "cooler",
        })
        result = _parse_llm_response(response, "I'm hot")
        assert result is not None
        assert result.action_type == "adjust"
        assert result.direction == "cooler"

    def test_markdown_fenced_json(self):
        response = '```json\n{"action_type": "set_temp", "target_temp": 77}\n```'
        result = _parse_llm_response(response, "set to 77")
        assert result is not None
        assert result.action_type == "set_temp"

    def test_invalid_json(self):
        result = _parse_llm_response("not json", "test")
        assert result is None

    def test_invalid_action_type(self):
        response = json.dumps({"action_type": "invalid_type"})
        result = _parse_llm_response(response, "test")
        assert result is None


# ── Integration: parse() with LLM ───────────────────────────────

class TestParseWithLLM:
    def test_llm_success(self):
        """When LLM returns valid JSON, use it."""
        def mock_llm(prompt):
            return json.dumps({
                "action_type": "set_temp",
                "target_temp": 77,
                "zone": "upstairs",
                "direction": None,
            }), True

        result = parse("please set the bedroom to 77 degrees", call_llm_fn=mock_llm)
        assert result.action_type == "set_temp"
        assert result.target_temp == 77
        assert result.zone == "upstairs"

    def test_llm_failure_falls_back_to_regex(self):
        """When LLM returns invalid response, fall back to regex."""
        def mock_llm(prompt):
            return "sorry I can't parse that", False

        result = parse("set to 77", call_llm_fn=mock_llm)
        assert result.action_type == "set_temp"
        assert result.target_temp == 77

    def test_llm_exception_falls_back_to_regex(self):
        """When LLM throws, fall back to regex."""
        def mock_llm(prompt):
            raise ConnectionError("LLM is down")

        result = parse("set to 77", call_llm_fn=mock_llm)
        assert result.action_type == "set_temp"
        assert result.target_temp == 77

    def test_no_llm_uses_regex(self):
        """When no LLM function provided, use regex."""
        result = parse("set to 77", call_llm_fn=None)
        assert result.action_type == "set_temp"
        assert result.target_temp == 77

    def test_empty_text(self):
        result = parse("", call_llm_fn=None)
        assert result.action_type == "unknown"

    def test_none_text(self):
        result = parse(None, call_llm_fn=None)
        assert result.action_type == "unknown"
