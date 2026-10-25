"""
Automated unit and integration test suite for the Travel Itinerary Planner.
Verifies schemas, clustering, LLM mock engine, recommendations, SQLite storage, and PDF output.
"""

import os
import sys
import uuid
import unittest
from datetime import date, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.schemas import (
    TripPreferences, ActivitySlot, DayItinerary, TripRecord, DestinationRecommendation
)
from services.clustering_service import (
    haversine_distance_km, estimate_transit_info, optimize_day_geographic_routing,
    score_clustering_efficiency
)
from services.llm_service import LLMService
from services.recommender_service import RecommenderService
from services.storage_service import (
    init_db, save_trip, get_all_trips, get_trip_by_id, delete_trip,
    export_trip_json, import_trip_json
)
from services.pdf_service import generate_itinerary_pdf


class TestTravelPlanner(unittest.TestCase):
    def setUp(self):
        init_db()
        self.prefs = TripPreferences(
            destination="Tokyo",
            duration_days=3,
            start_date=str(date.today()),
            end_date=str(date.today() + timedelta(days=2)),
            budget_level="Moderate",
            travel_party="Solo Traveler 🎒",
            pace="Moderate",
            interests=["Historical Landmarks", "Local Street Food"],
            dietary="Vegetarian"
        )

    def test_clustering_haversine(self):
        """Verify Haversine distance calculation and transit logic."""
        # Tokyo Station to Shibuya Crossing is ~6.5 km
        dist = haversine_distance_km(35.6812, 139.7671, 35.6598, 139.7024)
        self.assertGreater(dist, 5.0)
        self.assertLess(dist, 8.5)

        # Walking distance ~ 0.5 km
        short_desc, mins = estimate_transit_info(0.5)
        self.assertIn("stroll", short_desc)
        self.assertLessEqual(mins, 15)

    def test_llm_demo_generation(self):
        """Verify Demo Mode creates structured day itineraries with sequential slots."""
        days = LLMService.generate_itinerary(self.prefs, provider="demo")
        self.assertEqual(len(days), 3)
        for day in days:
            self.assertEqual(len(day.slots), 4)
            slot_ids = [s.slot_id for s in day.slots]
            self.assertEqual(slot_ids, ["morning", "lunch", "afternoon", "evening"])
            self.assertGreater(day.daily_cost_usd, 0)
            self.assertGreater(day.daily_cost_inr, 0)

    def test_slot_swap(self):
        """Verify dynamic slot swapping updates activity without wiping other slots."""
        days = LLMService.generate_itinerary(self.prefs, provider="demo")
        day1 = days[0]
        orig_slot = day1.slots[1]  # Lunch slot
        
        swapped = LLMService.swap_single_slot(
            target_slot=orig_slot,
            day_context=day1,
            swap_intent="Find a pure vegan cafe with organic smoothie bowls",
            preferences=self.prefs,
            provider="demo"
        )
        self.assertEqual(swapped.slot_id, "lunch")
        self.assertIn("Vegetarian", swapped.tags)
        self.assertNotEqual(swapped.title, orig_slot.title)

    def test_destination_recommender(self):
        """Verify Tab 1 destination recommender returns top 3 matching cities."""
        recs = RecommenderService.recommend_destinations(
            vibe_preference="Beach & Coastal Escapes 🏖️",
            budget_level="Budget",
            travel_party="Group of Friends 👥",
            travel_month="December"
        )
        self.assertEqual(len(recs), 3)
        for r in recs:
            self.assertIsInstance(r, DestinationRecommendation)
            self.assertGreaterEqual(r.vibe_match_score, 60)
            self.assertGreater(len(r.highlight_tags), 0)

    def test_sqlite_persistence_and_export(self):
        """Verify saving, loading, deleting trips and JSON export/import."""
        days = LLMService.generate_itinerary(self.prefs, provider="demo")
        test_id = f"test_{uuid.uuid4().hex[:6]}"
        trip = TripRecord(
            id=test_id,
            trip_name="Automated Test Trip",
            destination="Paris",
            duration_days=3,
            start_date=self.prefs.start_date,
            end_date=self.prefs.end_date,
            preferences=self.prefs,
            days=days
        )
        trip.recalculate_totals()
        
        # Save
        saved = save_trip(trip)
        self.assertTrue(saved)

        # Retrieve
        fetched = get_trip_by_id(test_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.destination, "Paris")
        self.assertEqual(len(fetched.days), 3)

        # JSON Export / Import
        json_str = export_trip_json(fetched)
        imported = import_trip_json(json_str)
        self.assertEqual(imported.id, test_id)

        # Delete
        deleted = delete_trip(test_id)
        self.assertTrue(deleted)
        self.assertIsNone(get_trip_by_id(test_id))

    def test_pdf_generation(self):
        """Verify ReportLab PDF generation builds a valid PDF binary."""
        days = LLMService.generate_itinerary(self.prefs, provider="demo")
        trip = TripRecord(
            id="pdf_test",
            trip_name="PDF Validation Expedition",
            destination="Tokyo",
            duration_days=3,
            start_date=self.prefs.start_date,
            end_date=self.prefs.end_date,
            preferences=self.prefs,
            days=days
        )
        trip.recalculate_totals()
        pdf_bytes = generate_itinerary_pdf(trip)
        self.assertIsInstance(pdf_bytes, bytes)
        # PDF files begin with %PDF
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_invalid_json_import(self):
        """Verify robust handling of invalid, empty, or corrupted JSON strings."""
        with self.assertRaises(ValueError):
            import_trip_json("")
        with self.assertRaises(ValueError):
            import_trip_json("{bad json")
        with self.assertRaises(ValueError):
            import_trip_json('{"some": "unrelated_data"}')
        with self.assertRaises(ValueError):
            import_trip_json(None)

    def test_live_api_failure_raises_error(self):
        """Verify live provider call without credentials raises RuntimeError and does not silently mock."""
        with self.assertRaises(RuntimeError):
            LLMService.generate_itinerary(self.prefs, provider="gemini", api_key="")
        with self.assertRaises(RuntimeError):
            LLMService.generate_itinerary(self.prefs, provider="groq", api_key="")

    def test_dietary_lunch_resolution(self):
        """Verify dietary preference is authentically reflected in lunch recommendations."""
        dietary_cases = ["Vegetarian", "Vegan", "Halal", "Jain"]
        for diet in dietary_cases:
            prefs = TripPreferences(
                destination="Tokyo",
                duration_days=1,
                start_date=str(date.today()),
                end_date=str(date.today()),
                budget_level="Moderate",
                travel_party="Solo Traveler 🎒",
                pace="Moderate",
                interests=["Local Street Food"],
                dietary=diet
            )
            days = LLMService.generate_itinerary(prefs, provider="demo")
            lunch_slot = days[0].slots[1]
            self.assertEqual(lunch_slot.slot_id, "lunch")
            self.assertIn(diet, lunch_slot.tags)

    def test_multi_day_extended_generation(self):
        """Verify multi-day (7 days) generation produces 4 complete slots for each day."""
        prefs_7 = TripPreferences(
            destination="Tokyo",
            duration_days=7,
            start_date=str(date.today()),
            end_date=str(date.today() + timedelta(days=6)),
            budget_level="Luxury",
            travel_party="Couples Getaway 💕",
            pace="Relaxed",
            interests=["Art, Design & Museums", "Nightlife & Rooftop Lounges"],
            dietary="Halal"
        )
        days = LLMService.generate_itinerary(prefs_7, provider="demo")
        self.assertEqual(len(days), 7)
        for d_idx, day in enumerate(days, 1):
            self.assertEqual(day.day_number, d_idx)
            self.assertEqual(len(day.slots), 4)

    def test_recommender_travel_party_scoring(self):
        """Verify travel party affects destination scoring."""
        recs_family = RecommenderService.recommend_destinations(
            vibe_preference="Mountains & Alpine Trails 🏔️",
            budget_level="Luxury",
            travel_party="Family with Kids 👨‍👩‍👧‍👦",
            travel_month="July"
        )
        top_destinations = [r.name for r in recs_family]
        self.assertIn("Zurich & Interlaken", top_destinations)

    def test_geographic_clustering_routing(self):
        """Verify geographic route optimization preserves all 4 activities and calculates transit."""
        days = LLMService.generate_itinerary(self.prefs, provider="demo")
        day = days[0]
        optimized = optimize_day_geographic_routing(day)
        self.assertEqual(len(optimized.slots), 4)
        self.assertEqual([s.slot_id for s in optimized.slots], ["morning", "lunch", "afternoon", "evening"])
        self.assertGreaterEqual(optimized.walking_distance_km, 0)
        clustering_eval = score_clustering_efficiency(optimized)
        self.assertIn("total_transit_km", clustering_eval)
        self.assertIn("rating", clustering_eval)


if __name__ == "__main__":
    unittest.main()
