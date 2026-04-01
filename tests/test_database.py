"""
Tests for database.py — SQLite operations, schema, queries.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta

import database


@pytest.fixture
def db():
    """Create a temporary database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(path)
    yield path
    os.unlink(path)


class TestDatabaseInit:
    def test_creates_tables(self, db):
        conn = database._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        conn.close()
        assert "climate_log" in table_names
        assert "decisions" in table_names
        assert "messages" in table_names
        assert "errors" in table_names
        assert "heartbeat" in table_names
        assert "manual_overrides" in table_names

    def test_heartbeat_initialized(self, db):
        heartbeat = database.get_heartbeat()
        assert heartbeat is None  # No cycle has run yet


class TestLogAndRetrieve:
    def test_log_and_get_decision(self, db):
        database.log_decision(
            indoor_temp=72.0, outdoor_temp=85.0,
            action="set_temperature", temperature=74.0,
            reasoning="Too warm", raw_response='{"action":"set_temperature"}',
            zone="Upstairs"
        )
        decisions = database.get_recent_decisions(10)
        assert len(decisions) == 1
        assert decisions[0]["action"] == "set_temperature"
        assert decisions[0]["temperature"] == 74.0

    # Regression: ISSUE-003 — zone column was missing from history query
    # Found by /qa on 2026-03-31
    def test_decision_includes_zone(self, db):
        database.log_decision(
            indoor_temp=72.0, outdoor_temp=85.0,
            action="set_temperature", temperature=74.0,
            reasoning="test", raw_response="{}",
            zone="Upstairs Bedroom"
        )
        decisions = database.get_recent_decisions(10)
        assert decisions[0]["zone"] == "Upstairs Bedroom"

    def test_log_and_get_message(self, db):
        database.log_message(12345, "set to 78", "Done")
        messages = database.get_recent_messages(10)
        assert len(messages) == 1
        assert messages[0]["text"] == "set to 78"
        assert messages[0]["agent_response"] == "Done"

    def test_log_climate(self, db):
        database.log_climate(
            indoor_temp=72.0, indoor_humidity=45.0,
            outdoor_temp=85.0, outdoor_humidity=50.0,
            forecast="sunny", hvac_mode="cooling",
            hvac_running=True, target_temp=72.0,
            action="no_change", reasoning="comfortable",
            zone="Kitchen"
        )
        entries = database.get_climate_log_since(
            datetime.utcnow() - timedelta(hours=1)
        )
        assert len(entries) == 1
        assert entries[0]["zone"] == "Kitchen"

    def test_log_error(self, db):
        database.log_error("llm", "timeout", "Server timed out")
        conn = database._get_conn()
        row = conn.execute("SELECT * FROM errors").fetchone()
        conn.close()
        assert row["component"] == "llm"
        assert row["error_type"] == "timeout"

    def test_manual_override(self, db):
        database.log_manual_override(75.0, 72.0, zone="Bedroom")
        override_time = database.get_last_manual_override_time()
        assert override_time is not None

    def test_heartbeat_update(self, db):
        database.update_heartbeat()
        heartbeat = database.get_heartbeat()
        assert heartbeat is not None


class TestRateLimiting:
    def test_count_changes_empty(self, db):
        count = database.count_temp_changes_last_hour()
        assert count == 0

    def test_count_changes_within_hour(self, db):
        for i in range(3):
            database.log_decision(
                indoor_temp=72, outdoor_temp=85,
                action="set_temperature", temperature=72 + i,
                reasoning="test", raw_response="{}"
            )
        count = database.count_temp_changes_last_hour()
        assert count == 3


class TestCleanup:
    def test_cleanup_old_records(self, db):
        # Insert an old record by manually setting timestamp
        conn = database._get_conn()
        old_ts = (datetime.utcnow() - timedelta(days=100)).isoformat()
        conn.execute(
            "INSERT INTO decisions (timestamp, action, reasoning, raw_response) VALUES (?, ?, ?, ?)",
            (old_ts, "no_change", "old", "{}")
        )
        conn.commit()
        conn.close()

        # Insert a fresh record
        database.log_decision(72, 85, "no_change", None, "fresh", "{}")

        database.cleanup_old_records(retention_days=90)
        decisions = database.get_recent_decisions(100)
        assert len(decisions) == 1
        assert decisions[0]["reasoning"] == "fresh"


class TestExport:
    def test_export_csv_no_data(self, db):
        result = database.export_climate_csv()
        assert result == "No data"

    def test_export_csv_with_data(self, db):
        database.log_climate(
            indoor_temp=72.0, indoor_humidity=45.0,
            outdoor_temp=85.0, outdoor_humidity=50.0,
            forecast="sunny", hvac_mode="cooling",
            hvac_running=False, target_temp=72.0,
            action="no_change", reasoning="ok"
        )
        csv = database.export_climate_csv()
        assert "indoor_temp" in csv
        assert "72.0" in csv
