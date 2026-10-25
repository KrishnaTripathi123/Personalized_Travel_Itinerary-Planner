"""
Simulated database persistence service using SQLite and JSON for offline portability.
Provides persistent storage, retrieval, deletion, and JSON export/import of trips.
"""

import os
import sqlite3
import json
import logging
from typing import List, Optional
from models.schemas import TripRecord, TripPreferences, DayItinerary, ActivitySlot

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "trips.db")


def init_db():
    """Initializes the SQLite database schema if not present."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trips (
                    id TEXT PRIMARY KEY,
                    trip_name TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    duration_days INTEGER NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    budget_level TEXT,
                    travel_party TEXT,
                    total_cost_usd REAL,
                    total_cost_inr REAL,
                    created_at TEXT,
                    trip_data_json TEXT NOT NULL
                )
            """)
    finally:
        conn.close()


def save_trip(trip: TripRecord) -> bool:
    """Saves or updates a trip record in SQLite."""
    try:
        init_db()
        payload = json.dumps(trip.to_dict())
        conn = sqlite3.connect(DB_PATH)
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trips (
                        id, trip_name, destination, duration_days, start_date, end_date,
                        budget_level, travel_party, total_cost_usd, total_cost_inr,
                        created_at, trip_data_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trip.id,
                    trip.trip_name,
                    trip.destination,
                    trip.duration_days,
                    trip.start_date,
                    trip.end_date,
                    trip.preferences.budget_level,
                    trip.preferences.travel_party,
                    trip.total_cost_usd,
                    trip.total_cost_inr,
                    trip.created_at,
                    payload
                ))
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to save trip to SQLite: {e}")
        return False


def get_all_trips() -> List[TripRecord]:
    """Retrieves all saved trips from SQLite ordered by creation time descending."""
    try:
        init_db()
        trips = []
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trip_data_json FROM trips ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            for row in rows:
                try:
                    data = json.loads(row[0])
                    trips.append(TripRecord.from_dict(data))
                except Exception as inner_e:
                    logger.warning(f"Error parsing trip row: {inner_e}")
            return trips
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to read trips from SQLite: {e}")
        return []


def get_trip_by_id(trip_id: str) -> Optional[TripRecord]:
    """Retrieves a single trip record by ID."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT trip_data_json FROM trips WHERE id = ?", (trip_id,))
            row = cursor.fetchone()
            if row:
                return TripRecord.from_dict(json.loads(row[0]))
            return None
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to get trip by ID: {e}")
        return None


def delete_trip(trip_id: str) -> bool:
    """Deletes a trip record by ID."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trips WHERE id = ?", (trip_id,))
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to delete trip: {e}")
        return False


def export_trip_json(trip: TripRecord) -> str:
    """Serializes a trip into formatted JSON string for export."""
    return json.dumps(trip.to_dict(), indent=2)


def import_trip_json(json_content: str) -> TripRecord:
    """Parses and validates JSON content into a TripRecord, persisting it to SQLite."""
    if not json_content or not isinstance(json_content, str) or not json_content.strip():
        raise ValueError("Trip JSON content cannot be empty.")
    
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as jde:
        raise ValueError(f"Invalid JSON format: {jde.msg} at line {jde.lineno}")

    if not isinstance(data, dict):
        raise ValueError("Imported JSON must contain a top-level trip object.")

    if "destination" not in data and "preferences" not in data and "trip_name" not in data:
        raise ValueError("Incompatible trip JSON: missing destination or trip details.")

    try:
        trip = TripRecord.from_dict(data)
    except Exception as schema_err:
        raise ValueError(f"Incompatible trip data schema: {schema_err}")

    if not trip.days:
        logger.warning(f"Imported trip '{trip.trip_name}' has no itinerary days.")

    save_trip(trip)
    return trip
