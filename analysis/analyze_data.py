#!/usr/bin/env python3
"""
Data analysis script for AI Thermostat comfort model tuning.
Joins manual_overrides and messages with climate_log to understand patterns.

Run: python analysis/analyze_data.py
"""

import sqlite3
import csv
import os
import sys
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "thermostat.db")


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def analyze_overrides(conn: sqlite3.Connection):
    """Join manual_overrides with nearest climate_log row."""
    print("\n" + "=" * 60)
    print("MANUAL OVERRIDES WITH CLIMATE CONTEXT")
    print("=" * 60)

    rows = conn.execute("""
        SELECT
            mo.timestamp,
            mo.zone,
            mo.detected_target,
            mo.previous_target,
            mo.detected_target - mo.previous_target AS delta,
            cl.indoor_temp,
            cl.outdoor_temp,
            cl.indoor_humidity,
            cl.hvac_mode,
            cl.hvac_running
        FROM manual_overrides mo
        LEFT JOIN climate_log cl ON cl.id = (
            SELECT id FROM climate_log
            WHERE timestamp <= mo.timestamp
            ORDER BY timestamp DESC LIMIT 1
        )
        ORDER BY mo.timestamp
    """).fetchall()

    if not rows:
        print("No manual overrides found.")
        return []

    print(f"\nTotal overrides: {len(rows)}")

    # Stats
    deltas_up = [r["delta"] for r in rows if r["delta"] and r["delta"] > 0]
    deltas_down = [r["delta"] for r in rows if r["delta"] and r["delta"] < 0]

    if deltas_up:
        print(f"Increased temp: {len(deltas_up)}x, avg +{sum(deltas_up)/len(deltas_up):.1f}F")
    if deltas_down:
        print(f"Decreased temp: {len(deltas_down)}x, avg {sum(deltas_down)/len(deltas_down):.1f}F")

    # By zone
    zones = set(r["zone"] for r in rows if r["zone"])
    for zone in sorted(zones):
        zone_rows = [r for r in rows if r["zone"] == zone]
        zone_deltas = [r["delta"] for r in zone_rows if r["delta"]]
        if zone_deltas:
            print(f"\n  {zone}: {len(zone_rows)} overrides, avg delta {sum(zone_deltas)/len(zone_deltas):+.1f}F")

    # By time of day
    print("\nBy time of day:")
    buckets = {"morning (6-12)": [], "afternoon (12-18)": [],
               "evening (18-22)": [], "night (22-6)": []}
    for r in rows:
        try:
            hour = datetime.fromisoformat(r["timestamp"]).hour
        except (ValueError, TypeError):
            continue
        if 6 <= hour < 12:
            buckets["morning (6-12)"].append(r)
        elif 12 <= hour < 18:
            buckets["afternoon (12-18)"].append(r)
        elif 18 <= hour < 22:
            buckets["evening (18-22)"].append(r)
        else:
            buckets["night (22-6)"].append(r)

    for bucket, bucket_rows in buckets.items():
        if bucket_rows:
            deltas = [r["delta"] for r in bucket_rows if r["delta"]]
            avg = sum(deltas) / len(deltas) if deltas else 0
            print(f"  {bucket}: {len(bucket_rows)} overrides, avg delta {avg:+.1f}F")

    # Detail table
    print(f"\n{'Timestamp':<20} {'Zone':<22} {'Prev':>5} {'New':>5} {'Delta':>6} {'Indoor':>7} {'Outdoor':>8} {'Mode':<8}")
    print("-" * 100)
    for r in rows:
        print(f"{(r['timestamp'] or '')[:19]:<20} {(r['zone'] or 'N/A'):<22} "
              f"{r['previous_target'] or 0:5.0f} {r['detected_target'] or 0:5.0f} "
              f"{r['delta'] or 0:+6.1f} {r['indoor_temp'] or 0:7.1f} "
              f"{r['outdoor_temp'] or 0:8.1f} {(r['hvac_mode'] or 'N/A'):<8}")

    return [dict(r) for r in rows]


def analyze_messages(conn: sqlite3.Connection):
    """Parse messages for temperature requests and join with climate context."""
    print("\n" + "=" * 60)
    print("USER MESSAGES WITH CLIMATE CONTEXT")
    print("=" * 60)

    rows = conn.execute("""
        SELECT
            m.timestamp,
            m.text,
            m.agent_response,
            cl.indoor_temp,
            cl.outdoor_temp,
            cl.hvac_mode,
            cl.target_temp
        FROM messages m
        LEFT JOIN climate_log cl ON cl.id = (
            SELECT id FROM climate_log
            WHERE timestamp <= m.timestamp
            ORDER BY timestamp DESC LIMIT 1
        )
        ORDER BY m.timestamp
    """).fetchall()

    if not rows:
        print("No messages found.")
        return

    print(f"\nTotal messages: {len(rows)}")

    # Categorize messages
    import re
    temp_requests = []
    questions = []
    complaints = []
    other = []

    for r in rows:
        text = (r["text"] or "").lower()
        if re.search(r"set.*\d{2}", text) or re.search(r"\d{2}.*degree", text):
            temp_requests.append(r)
        elif "?" in text or any(w in text for w in ["what", "how", "when", "why"]):
            questions.append(r)
        elif any(w in text for w in ["hot", "cold", "warm", "cool", "freezing"]):
            complaints.append(r)
        else:
            other.append(r)

    print(f"  Temperature requests: {len(temp_requests)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Comfort complaints: {len(complaints)}")
    print(f"  Other: {len(other)}")

    if temp_requests:
        print("\nTemperature requests:")
        for r in temp_requests:
            print(f"  [{(r['timestamp'] or '')[:16]}] \"{r['text']}\" "
                  f"(indoor={r['indoor_temp'] or '?'}F, outdoor={r['outdoor_temp'] or '?'}F)")

    if complaints:
        print("\nComfort complaints:")
        for r in complaints:
            print(f"  [{(r['timestamp'] or '')[:16]}] \"{r['text']}\" "
                  f"(indoor={r['indoor_temp'] or '?'}F, outdoor={r['outdoor_temp'] or '?'}F)")


def analyze_zone_differential(conn: sqlite3.Connection):
    """Compute upstairs vs downstairs temperature differential."""
    print("\n" + "=" * 60)
    print("ZONE DIFFERENTIAL ANALYSIS")
    print("=" * 60)

    rows = conn.execute("""
        SELECT
            c1.timestamp,
            c1.indoor_temp AS upstairs_temp,
            c2.indoor_temp AS downstairs_temp,
            c1.indoor_temp - c2.indoor_temp AS differential,
            c1.outdoor_temp
        FROM climate_log c1
        JOIN climate_log c2 ON c2.timestamp = c1.timestamp
            AND c2.zone != c1.zone
        WHERE c1.zone LIKE '%Upstairs%'
          AND c2.zone LIKE '%Downstairs%'
          AND c1.indoor_temp IS NOT NULL
          AND c2.indoor_temp IS NOT NULL
        ORDER BY c1.timestamp
    """).fetchall()

    if not rows:
        # Try alternate: match by closest timestamps
        rows = conn.execute("""
            SELECT
                c1.timestamp,
                c1.indoor_temp AS upstairs_temp,
                c2.indoor_temp AS downstairs_temp,
                c1.indoor_temp - c2.indoor_temp AS differential,
                c1.outdoor_temp
            FROM climate_log c1
            JOIN climate_log c2 ON c2.id = (
                SELECT id FROM climate_log
                WHERE zone LIKE '%Downstairs%'
                  AND ABS(julianday(timestamp) - julianday(c1.timestamp)) < 0.01
                ORDER BY ABS(julianday(timestamp) - julianday(c1.timestamp))
                LIMIT 1
            )
            WHERE c1.zone LIKE '%Upstairs%'
              AND c1.indoor_temp IS NOT NULL
            ORDER BY c1.timestamp
        """).fetchall()

    if not rows:
        print("Not enough multi-zone data to compute differential.")
        return

    diffs = [r["differential"] for r in rows if r["differential"] is not None]
    if diffs:
        print(f"\nData points: {len(diffs)}")
        print(f"Avg differential (upstairs - downstairs): {sum(diffs)/len(diffs):+.1f}F")
        print(f"Min: {min(diffs):+.1f}F, Max: {max(diffs):+.1f}F")

        # By outdoor temp range
        print("\nBy outdoor temp:")
        for low, high, label in [(0, 65, "Cold (<65F)"), (65, 80, "Mild (65-80F)"),
                                  (80, 95, "Hot (80-95F)"), (95, 200, "Extreme (95F+)")]:
            bucket = [r["differential"] for r in rows
                     if r["outdoor_temp"] and low <= r["outdoor_temp"] < high
                     and r["differential"] is not None]
            if bucket:
                print(f"  {label}: avg {sum(bucket)/len(bucket):+.1f}F ({len(bucket)} pts)")


def analyze_thermal_drift(conn: sqlite3.Connection):
    """Estimate thermal drift rate when HVAC is off."""
    print("\n" + "=" * 60)
    print("THERMAL DRIFT ANALYSIS")
    print("=" * 60)

    rows = conn.execute("""
        SELECT timestamp, zone, indoor_temp, outdoor_temp, hvac_running
        FROM climate_log
        WHERE indoor_temp IS NOT NULL
        ORDER BY zone, timestamp
    """).fetchall()

    if len(rows) < 2:
        print("Not enough data for drift analysis.")
        return

    # Group consecutive off periods by zone
    zones = {}
    for r in rows:
        zone = r["zone"] or "default"
        if zone not in zones:
            zones[zone] = []
        zones[zone].append(r)

    for zone, zone_rows in zones.items():
        drift_rates = []
        for i in range(1, len(zone_rows)):
            prev, curr = zone_rows[i - 1], zone_rows[i]
            # Both must be HVAC off
            if prev["hvac_running"] or curr["hvac_running"]:
                continue
            try:
                t0 = datetime.fromisoformat(prev["timestamp"])
                t1 = datetime.fromisoformat(curr["timestamp"])
                dt_hours = (t1 - t0).total_seconds() / 3600
                if dt_hours <= 0 or dt_hours > 2:  # Skip gaps > 2 hours
                    continue
                dt_temp = curr["indoor_temp"] - prev["indoor_temp"]
                drift_rates.append(dt_temp / dt_hours)
            except (ValueError, TypeError):
                continue

        if drift_rates:
            avg = sum(drift_rates) / len(drift_rates)
            print(f"\n{zone}:")
            print(f"  Samples: {len(drift_rates)}")
            print(f"  Avg drift: {avg:+.2f}F/hour")
            print(f"  Range: {min(drift_rates):+.2f} to {max(drift_rates):+.2f}F/hour")


def export_training_csv(overrides: list, db_path: str):
    """Export labeled training data as CSV."""
    csv_path = os.path.join(os.path.dirname(db_path), "analysis", "training_data.csv")
    if not overrides:
        print(f"\nNo data to export.")
        return

    fieldnames = ["timestamp", "zone", "previous_target", "detected_target", "delta",
                  "indoor_temp", "outdoor_temp", "indoor_humidity", "hvac_mode", "hvac_running"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in overrides:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"\nTraining data exported to: {csv_path}")


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    db_path = os.path.abspath(db_path)

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("Usage: python analysis/analyze_data.py [path/to/thermostat.db]")
        sys.exit(1)

    print(f"Analyzing: {db_path}")
    conn = get_conn(db_path)

    try:
        # Table counts
        print("\nTable row counts:")
        for table in ["climate_log", "decisions", "messages", "manual_overrides", "location_log"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {count}")
            except sqlite3.OperationalError:
                print(f"  {table}: (table not found)")

        overrides = analyze_overrides(conn)
        analyze_messages(conn)
        analyze_zone_differential(conn)
        analyze_thermal_drift(conn)
        export_training_csv(overrides, db_path)

    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
