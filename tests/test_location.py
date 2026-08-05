"""
Tests for location.py — ARP-based presence detection and vacation mode logic.
"""

import pytest
from unittest.mock import patch, AsyncMock

import location


@pytest.fixture(autouse=True)
def reset_location_state():
    """Reset module-level state before each test."""
    location._enabled = True
    location._phone_macs = ["aa:bb:cc:dd:ee:ff", "11:22:33:44:55:66"]
    location._vacation_mode = False
    location._vacation_change_callback = None
    location._initialized = True
    location._vacation_temp_min = 60
    location._vacation_temp_max = 85
    location._vacation_eval_interval_minutes = 10
    location._consecutive_misses = 0
    location._required_misses = 2
    yield


# ── Haversine Tests ──────────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        """Distance from a point to itself is 0."""
        assert location._haversine(37.9324, -121.6894, 37.9324, -121.6894) == 0.0

    def test_known_distance_sf_to_la(self):
        """SF to LA is roughly 347 miles."""
        dist = location._haversine(37.7749, -122.4194, 34.0522, -118.2437)
        assert 340 < dist < 360, f"SF-LA should be ~347mi, got {dist:.1f}"

    def test_known_distance_ny_to_london(self):
        """NYC to London is roughly 3459 miles."""
        dist = location._haversine(40.7128, -74.0060, 51.5074, -0.1278)
        assert 3400 < dist < 3520, f"NYC-London should be ~3459mi, got {dist:.1f}"

    def test_short_distance(self):
        """A small offset should give a short distance (< 1 mile)."""
        dist = location._haversine(37.9324, -121.6894, 37.9424, -121.6894)
        assert 0.5 < dist < 1.0, f"Short distance should be ~0.69mi, got {dist:.3f}"

    def test_symmetric(self):
        """Distance A→B should equal B→A."""
        d1 = location._haversine(37.7749, -122.4194, 34.0522, -118.2437)
        d2 = location._haversine(34.0522, -118.2437, 37.7749, -122.4194)
        assert abs(d1 - d2) < 0.01


# ── ARP Presence Detection Tests ────────────────────────────────

ARP_OUTPUT_WITH_BOTH = """\
  192.168.1.1           00-11-22-33-44-55     dynamic
  192.168.1.50          aa-bb-cc-dd-ee-ff     dynamic
  192.168.1.51          11-22-33-44-55-66     dynamic
  192.168.1.200         ff-ee-dd-cc-bb-aa     dynamic
"""

ARP_OUTPUT_WITH_ONE = """\
  192.168.1.1           00-11-22-33-44-55     dynamic
  192.168.1.50          aa-bb-cc-dd-ee-ff     dynamic
  192.168.1.200         ff-ee-dd-cc-bb-aa     dynamic
"""

ARP_OUTPUT_WITH_NONE = """\
  192.168.1.1           00-11-22-33-44-55     dynamic
  192.168.1.200         ff-ee-dd-cc-bb-aa     dynamic
"""


class TestARPPresence:
    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_mac_found_and_reachable_vacation_off(self, mock_arp, mock_ping):
        """MAC found in ARP + ping succeeds → vacation stays OFF."""
        mock_arp.return_value = ARP_OUTPUT_WITH_BOTH
        location.check_arp_presence()
        assert location.is_vacation_mode() is False
        assert location._consecutive_misses == 0

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_no_mac_increments_miss_counter(self, mock_arp, mock_ping):
        """No MACs in ARP table → miss counter increments."""
        mock_arp.return_value = ARP_OUTPUT_WITH_NONE
        location.check_arp_presence()
        assert location._consecutive_misses == 1
        assert location.is_vacation_mode() is False

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_two_consecutive_misses_activates_vacation(self, mock_arp, mock_ping):
        """2 consecutive misses → vacation ON."""
        mock_arp.return_value = ARP_OUTPUT_WITH_NONE

        location.check_arp_presence()  # miss 1
        assert location.is_vacation_mode() is False

        location.check_arp_presence()  # miss 2
        assert location.is_vacation_mode() is True

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_detection_after_vacation_deactivates_immediately(self, mock_arp, mock_ping):
        """Detection after vacation → immediate deactivation + counter reset."""
        mock_arp.return_value = ARP_OUTPUT_WITH_NONE
        location.check_arp_presence()
        location.check_arp_presence()
        assert location.is_vacation_mode() is True

        mock_arp.return_value = ARP_OUTPUT_WITH_BOTH
        location.check_arp_presence()
        assert location.is_vacation_mode() is False
        assert location._consecutive_misses == 0

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_partial_match_vacation_off(self, mock_arp, mock_ping):
        """One MAC found + reachable, one not in ARP → vacation OFF."""
        mock_arp.return_value = ARP_OUTPUT_WITH_ONE
        location.check_arp_presence()
        assert location.is_vacation_mode() is False
        assert location._consecutive_misses == 0

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_miss_then_detection_resets_counter(self, mock_arp, mock_ping):
        """One miss, then detection → counter resets, vacation stays OFF."""
        mock_arp.return_value = ARP_OUTPUT_WITH_NONE
        location.check_arp_presence()
        assert location._consecutive_misses == 1

        mock_arp.return_value = ARP_OUTPUT_WITH_BOTH
        location.check_arp_presence()
        assert location._consecutive_misses == 0
        assert location.is_vacation_mode() is False

    @patch("location._run_arp")
    def test_disabled_does_nothing(self, mock_arp):
        """When disabled, check_arp_presence does nothing."""
        location._enabled = False
        location.check_arp_presence()
        mock_arp.assert_not_called()

    @patch("location._run_arp")
    def test_arp_failure_does_not_crash(self, mock_arp):
        """ARP command failure doesn't crash or change state."""
        mock_arp.side_effect = Exception("subprocess failed")
        location.check_arp_presence()
        assert location._consecutive_misses == 0
        assert location.is_vacation_mode() is False

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_mac_matching_case_insensitive(self, mock_arp, mock_ping):
        """MAC matching is case-insensitive."""
        arp_upper = "  192.168.1.50          AA-BB-CC-DD-EE-FF     dynamic\n"
        mock_arp.return_value = arp_upper
        location.check_arp_presence()
        assert location._consecutive_misses == 0
        assert location.is_vacation_mode() is False

    @patch("location._ping_check", return_value=False)
    @patch("location._run_arp")
    def test_stale_arp_entry_ping_fails_counts_as_miss(self, mock_arp, mock_ping):
        """MAC in ARP cache but ping fails → stale entry, counts as miss."""
        mock_arp.return_value = ARP_OUTPUT_WITH_BOTH
        location.check_arp_presence()
        assert location._consecutive_misses == 1
        assert location.is_vacation_mode() is False

    @patch("location._ping_check", return_value=False)
    @patch("location._run_arp")
    def test_stale_entries_two_misses_activates_vacation(self, mock_arp, mock_ping):
        """MAC in ARP but unreachable for 2 cycles → vacation ON."""
        mock_arp.return_value = ARP_OUTPUT_WITH_BOTH
        location.check_arp_presence()  # miss 1 (ping fails)
        location.check_arp_presence()  # miss 2 (ping fails)
        assert location.is_vacation_mode() is True


# ── Callback Tests ───────────────────────────────────────────────

class TestVacationCallback:
    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_callback_fires_on_vacation_activation(self, mock_arp, mock_ping):
        """Callback fires when vacation mode activates."""
        callback = AsyncMock()
        location.set_vacation_change_callback(callback)

        mock_arp.return_value = ARP_OUTPUT_WITH_NONE
        location.check_arp_presence()  # miss 1
        callback.assert_not_called()

        location.check_arp_presence()  # miss 2 → vacation ON
        callback.assert_called_once_with(True)

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_callback_does_not_fire_when_no_change(self, mock_arp, mock_ping):
        """Callback does NOT fire when vacation mode doesn't change."""
        callback = AsyncMock()
        location.set_vacation_change_callback(callback)

        mock_arp.return_value = ARP_OUTPUT_WITH_BOTH
        location.check_arp_presence()
        callback.assert_not_called()

    @patch("location._ping_check", return_value=True)
    @patch("location._run_arp")
    def test_callback_fires_on_deactivation(self, mock_arp, mock_ping):
        """Callback fires when vacation mode turns OFF (return home)."""
        location._vacation_mode = True
        location._consecutive_misses = 5

        callback = AsyncMock()
        location.set_vacation_change_callback(callback)

        mock_arp.return_value = ARP_OUTPUT_WITH_BOTH
        location.check_arp_presence()
        callback.assert_called_once_with(False)


# ── Parse ARP IP Tests ───────────────────────────────────────────

class TestParseArpIp:
    def test_finds_ip_for_mac(self):
        """Correctly extracts IP for a known MAC."""
        ip = location._parse_arp_ip_for_mac(ARP_OUTPUT_WITH_BOTH, "aa:bb:cc:dd:ee:ff")
        assert ip == "192.168.1.50"

    def test_returns_none_for_unknown_mac(self):
        """Returns None when MAC is not in ARP output."""
        ip = location._parse_arp_ip_for_mac(ARP_OUTPUT_WITH_NONE, "aa:bb:cc:dd:ee:ff")
        assert ip is None

    def test_handles_dash_format(self):
        """Works with dash-separated MACs in ARP output."""
        ip = location._parse_arp_ip_for_mac(ARP_OUTPUT_WITH_BOTH, "11-22-33-44-55-66")
        assert ip == "192.168.1.51"


# ── Init Tests ───────────────────────────────────────────────────

class TestInit:
    def test_disabled_when_no_macs(self):
        """When no phone_macs configured, presence detection is disabled."""
        location._enabled = False
        location._initialized = False
        location._vacation_mode = False
        assert location.is_vacation_mode() is False

    def test_init_with_config(self):
        """init_location reads presence config correctly."""
        config = {
            "presence": {
                "phone_macs": ["AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66"],
                "consecutive_misses": 3,
            },
        }
        location.init_location(config)

        assert location._enabled is True
        assert location._phone_macs == ["aa:bb:cc:dd:ee:ff", "11:22:33:44:55:66"]
        assert location._required_misses == 3
        assert location._consecutive_misses == 0

    def test_init_disabled_no_macs(self):
        """init_location with empty phone_macs disables detection."""
        config = {
            "presence": {"phone_macs": []},
        }
        location.init_location(config)
        assert location._enabled is False

    def test_init_no_presence_section(self):
        """init_location with no presence section disables detection."""
        config = {}
        location.init_location(config)
        assert location._enabled is False

    def test_init_vacation_settings_from_presence(self):
        """Vacation settings can come from presence config."""
        config = {
            "presence": {
                "phone_macs": ["AA:BB:CC:DD:EE:FF"],
                "vacation_temp_min": 55,
                "vacation_temp_max": 90,
                "vacation_eval_interval_minutes": 120,
            },
        }
        location.init_location(config)
        assert location._vacation_temp_min == 55
        assert location._vacation_temp_max == 90
        assert location._vacation_eval_interval_minutes == 120

    def test_getters(self):
        """Getter functions return correct values."""
        location._vacation_temp_min = 55
        location._vacation_temp_max = 90
        location._vacation_eval_interval_minutes = 120
        location._vacation_mode = True

        assert location.get_vacation_temp_min() == 55
        assert location.get_vacation_temp_max() == 90
        assert location.get_vacation_eval_interval_minutes() == 120
        assert location.is_vacation_mode() is True
