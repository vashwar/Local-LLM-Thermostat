#!/usr/bin/env python3
"""
SQLite database module for the AI Thermostat Agent.
Handles climate logging, decision tracking, message history, and error logging.
"""

import sqlite3
import csv
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_db_path: Optional[str] = None


def _get_conn() -> sqlite3.Connection:
    """Get a database connection with WAL mode and 5s timeout."""
    conn = sqlite3.connect(_db_path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str):
    """Initialize the database and create tables if needed."""
    global _db_path
    _db_path = db_path

    conn = _get_conn()
    try:
        # climate_log — training data, kept indefinitely
        conn.execute("""
            CREATE TABLE IF NOT EXISTS climate_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                zone TEXT DEFAULT 'default',
                indoor_temp REAL,
                indoor_humidity REAL,
                outdoor_temp REAL,
                outdoor_humidity REAL,
                forecast TEXT,
                hvac_mode TEXT,
                hvac_running INTEGER,
                target_temp REAL,
                action TEXT,
                reasoning TEXT,
                user_comment TEXT
            )
        """)
        # Migration: add zone column if missing (existing DB)
        try:
            conn.execute("ALTER TABLE climate_log ADD COLUMN zone TEXT DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_climate_timestamp
            ON climate_log(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_climate_hvac_running
            ON climate_log(hvac_running)
        """)

        # decisions — 90-day retention
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                zone TEXT DEFAULT 'default',
                indoor_temp REAL,
                outdoor_temp REAL,
                action TEXT,
                temperature REAL,
                reasoning TEXT,
                raw_response TEXT
            )
        """)
        try:
            conn.execute("ALTER TABLE decisions ADD COLUMN zone TEXT DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_timestamp
            ON decisions(timestamp)
        """)

        # messages — 90-day retention
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                chat_id INTEGER,
                text TEXT,
                agent_response TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON messages(timestamp)
        """)

        # errors — 90-day retention
        conn.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                component TEXT,
                error_type TEXT,
                details TEXT
            )
        """)

        # heartbeat — single row, last successful cycle
        conn.execute("""
            CREATE TABLE IF NOT EXISTS heartbeat (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_cycle TEXT
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO heartbeat (id, last_cycle) VALUES (1, NULL)
        """)

        # manual_overrides — timestamps of detected manual changes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                zone TEXT DEFAULT 'default',
                detected_target REAL,
                previous_target REAL
            )
        """)
        try:
            conn.execute("ALTER TABLE manual_overrides ADD COLUMN zone TEXT DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass

        conn.commit()
        logger.info("Database initialized: %s", db_path)
    finally:
        conn.close()


def log_climate(indoor_temp: float, indoor_humidity: float,
                outdoor_temp: float, outdoor_humidity: float,
                forecast: str, hvac_mode: str, hvac_running: bool,
                target_temp: float, action: str, reasoning: str,
                user_comment: Optional[str] = None, zone: str = "default"):
    """Log a climate snapshot (training data — kept indefinitely)."""
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO climate_log
            (zone, indoor_temp, indoor_humidity, outdoor_temp, outdoor_humidity,
             forecast, hvac_mode, hvac_running, target_temp, action, reasoning, user_comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (zone, indoor_temp, indoor_humidity, outdoor_temp, outdoor_humidity,
              forecast, hvac_mode, int(hvac_running), target_temp, action, reasoning,
              user_comment))
        conn.commit()
    finally:
        conn.close()


def log_decision(indoor_temp: float, outdoor_temp: float,
                 action: str, temperature: Optional[float],
                 reasoning: str, raw_response: str, zone: str = "default"):
    """Log an agent decision (90-day retention)."""
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO decisions
            (zone, indoor_temp, outdoor_temp, action, temperature, reasoning, raw_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (zone, indoor_temp, outdoor_temp, action, temperature, reasoning, raw_response))
        conn.commit()
    finally:
        conn.close()


def log_message(chat_id: int, text: str, agent_response: Optional[str] = None):
    """Log a Telegram message exchange (90-day retention)."""
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO messages (chat_id, text, agent_response)
            VALUES (?, ?, ?)
        """, (chat_id, text, agent_response))
        conn.commit()
    finally:
        conn.close()


def log_error(component: str, error_type: str, details: str):
    """Log an error (90-day retention)."""
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO errors (component, error_type, details)
            VALUES (?, ?, ?)
        """, (component, error_type, details))
        conn.commit()
    finally:
        conn.close()


def log_manual_override(detected_target: float, previous_target: float,
                        zone: str = "default"):
    """Log a detected manual override."""
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO manual_overrides (zone, detected_target, previous_target)
            VALUES (?, ?, ?)
        """, (zone, detected_target, previous_target))
        conn.commit()
    finally:
        conn.close()


def get_recent_messages(limit: int = 10) -> list:
    """Get the most recent Telegram messages."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT timestamp, chat_id, text, agent_response
            FROM messages ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def get_recent_decisions(limit: int = 10) -> list:
    """Get the most recent agent decisions."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT timestamp, zone, indoor_temp, outdoor_temp, action, temperature, reasoning
            FROM decisions ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def count_temp_changes_last_hour() -> int:
    """Count temperature changes in the last hour (for rate limiting)."""
    conn = _get_conn()
    try:
        cutoff = (datetime.utcnow() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM decisions
            WHERE action = 'set_temperature' AND timestamp > ?
        """, (cutoff,)).fetchone()
        return row["cnt"]
    finally:
        conn.close()


def get_last_manual_override_time() -> Optional[datetime]:
    """Get the timestamp of the most recent manual override."""
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT timestamp FROM manual_overrides
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        if row:
            return datetime.fromisoformat(row["timestamp"])
        return None
    finally:
        conn.close()


def get_climate_log_since(since: datetime) -> list:
    """Get climate log entries since a given time (for weekly reports)."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT * FROM climate_log WHERE timestamp > ?
            ORDER BY timestamp ASC
        """, (since.strftime("%Y-%m-%d %H:%M:%S"),)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def export_climate_csv() -> str:
    """Export all climate_log data as CSV string."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM climate_log ORDER BY timestamp ASC").fetchall()
        if not rows:
            return "No data"
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(tuple(row))
        return output.getvalue()
    finally:
        conn.close()


def update_heartbeat():
    """Update the heartbeat with current timestamp."""
    conn = _get_conn()
    try:
        conn.execute("""
            UPDATE heartbeat SET last_cycle = datetime('now') WHERE id = 1
        """)
        conn.commit()
    finally:
        conn.close()


def get_heartbeat() -> Optional[str]:
    """Get the last heartbeat timestamp."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT last_cycle FROM heartbeat WHERE id = 1").fetchone()
        return row["last_cycle"] if row else None
    finally:
        conn.close()


def cleanup_old_records(retention_days: int = 90):
    """Delete records older than retention_days from 90-day tables."""
    cutoff = (datetime.utcnow() - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_conn()
    try:
        for table in ["decisions", "messages", "errors", "manual_overrides"]:
            conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
        conn.commit()
        logger.info("Cleaned up records older than %d days", retention_days)
    finally:
        conn.close()
