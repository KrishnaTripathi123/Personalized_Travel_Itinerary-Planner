"""
Data schemas and data transfer objects for Trip Preferences, Itineraries, and Destinations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime, date


@dataclass
class TripPreferences:
    destination: str
    duration_days: int
    start_date: str
    end_date: str
    budget_level: str  # Budget, Moderate, Luxury
    travel_party: str  # Solo, Couple, Family, Friends, Business
    pace: str          # Relaxed, Moderate, Fast-Paced
    interests: List[str] = field(default_factory=list)
    open_vibe: Optional[str] = None
    dietary: Optional[str] = "Any / Flexible"
    currency: str = "USD"
    custom_daily_budget_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TripPreferences":
        return cls(
            destination=data.get("destination", "Tokyo"),
            duration_days=int(data.get("duration_days", 3)),
            start_date=data.get("start_date", str(date.today())),
            end_date=data.get("end_date", str(date.today())),
            budget_level=data.get("budget_level", "Moderate"),
            travel_party=data.get("travel_party", "Solo Traveler 🎒"),
            pace=data.get("pace", "Moderate"),
            interests=data.get("interests", ["Historical Landmarks", "Local Street Food"]),
            open_vibe=data.get("open_vibe"),
            dietary=data.get("dietary", "Any / Flexible"),
            currency=data.get("currency", "USD"),
            custom_daily_budget_usd=data.get("custom_daily_budget_usd")
        )


@dataclass
class ActivitySlot:
    slot_id: str          # morning, lunch, afternoon, evening
    slot_title: str       # Morning Sightseeing, etc.
    time_window: str      # 09:00 - 12:30
    title: str
    description: str
    location_name: str
    latitude: float
    longitude: float
    zone: str
    category: str         # Sightseeing, Dining, Leisure, Culture, Nightlife
    estimated_cost_usd: float
    estimated_cost_inr: float
    transit_info: str = "10-15 mins nearby walking/transit"
    distance_km: float = 1.2
    tips: str = "Arrive early to avoid queues"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivitySlot":
        return cls(
            slot_id=data.get("slot_id", "morning"),
            slot_title=data.get("slot_title", "Activity Slot"),
            time_window=data.get("time_window", "09:00 - 12:00"),
            title=data.get("title", "Local Exploration"),
            description=data.get("description", "Discover the iconic local neighborhood."),
            location_name=data.get("location_name", "City Center"),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            zone=data.get("zone", "Central District"),
            category=data.get("category", "Sightseeing"),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 20.0)),
            estimated_cost_inr=float(data.get("estimated_cost_inr", 1700.0)),
            transit_info=data.get("transit_info", "Short walk"),
            distance_km=float(data.get("distance_km", 1.0)),
            tips=data.get("tips", "Comfortable walking shoes recommended."),
            tags=data.get("tags", [])
        )


@dataclass
class DayItinerary:
    day_number: int
    date_str: str
    theme: str
    zone_cluster: str
    slots: List[ActivitySlot] = field(default_factory=list)
    daily_cost_usd: float = 0.0
    daily_cost_inr: float = 0.0
    walking_distance_km: float = 0.0

    def recalculate_totals(self, usd_to_inr: float = 85.0):
        self.daily_cost_usd = sum(s.estimated_cost_usd for s in self.slots)
        self.daily_cost_inr = round(self.daily_cost_usd * usd_to_inr, 2)
        self.walking_distance_km = round(sum(s.distance_km for s in self.slots), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_number": self.day_number,
            "date_str": self.date_str,
            "theme": self.theme,
            "zone_cluster": self.zone_cluster,
            "slots": [s.to_dict() for s in self.slots],
            "daily_cost_usd": self.daily_cost_usd,
            "daily_cost_inr": self.daily_cost_inr,
            "walking_distance_km": self.walking_distance_km
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DayItinerary":
        slots = [ActivitySlot.from_dict(s) for s in data.get("slots", [])]
        inst = cls(
            day_number=int(data.get("day_number", 1)),
            date_str=data.get("date_str", "Day 1"),
            theme=data.get("theme", "Highlights Discovery"),
            zone_cluster=data.get("zone_cluster", "Central Hub"),
            slots=slots,
            daily_cost_usd=float(data.get("daily_cost_usd", 0.0)),
            daily_cost_inr=float(data.get("daily_cost_inr", 0.0)),
            walking_distance_km=float(data.get("walking_distance_km", 0.0))
        )
        if not inst.daily_cost_usd and slots:
            inst.recalculate_totals()
        elif (not inst.daily_cost_inr or inst.daily_cost_inr <= 0) and inst.daily_cost_usd > 0:
            inst.daily_cost_inr = round(inst.daily_cost_usd * 85.0, 2)
        return inst


@dataclass
class TripRecord:
    id: str
    trip_name: str
    destination: str
    duration_days: int
    start_date: str
    end_date: str
    preferences: TripPreferences
    days: List[DayItinerary] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_cost_inr: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    def recalculate_totals(self, usd_to_inr: float = 85.0):
        for d in self.days:
            d.recalculate_totals(usd_to_inr)
        self.total_cost_usd = sum(d.daily_cost_usd for d in self.days)
        self.total_cost_inr = round(self.total_cost_usd * usd_to_inr, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trip_name": self.trip_name,
            "destination": self.destination,
            "duration_days": self.duration_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "preferences": self.preferences.to_dict(),
            "days": [d.to_dict() for d in self.days],
            "total_cost_usd": self.total_cost_usd,
            "total_cost_inr": self.total_cost_inr,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TripRecord":
        prefs = TripPreferences.from_dict(data.get("preferences", {}))
        days = [DayItinerary.from_dict(d) for d in data.get("days", [])]
        inst = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            trip_name=data.get("trip_name", f"{prefs.destination} Odyssey"),
            destination=data.get("destination", prefs.destination),
            duration_days=int(data.get("duration_days", prefs.duration_days)),
            start_date=data.get("start_date", prefs.start_date),
            end_date=data.get("end_date", prefs.end_date),
            preferences=prefs,
            days=days,
            total_cost_usd=float(data.get("total_cost_usd", 0.0)),
            total_cost_inr=float(data.get("total_cost_inr", 0.0)),
            created_at=data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
        )
        if not inst.total_cost_usd and days:
            inst.recalculate_totals()
        elif (not inst.total_cost_inr or inst.total_cost_inr <= 0) and inst.total_cost_usd > 0:
            inst.total_cost_inr = round(inst.total_cost_usd * 85.0, 2)
        return inst


@dataclass
class DestinationRecommendation:
    name: str
    country: str
    vibe_match_score: int       # 0 - 100%
    highlight_tags: List[str]
    estimated_daily_usd: float
    estimated_daily_inr: float
    best_season: str
    why_recommended: str
    top_attractions: List[str]
    weather_tag: str = "Pleasant & Clear"
    safety_badge: str = "High Tourist Safety"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
